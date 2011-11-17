#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define CHKERR(err, str) \
	if (err != CL_SUCCESS) \
	{ \
		fprintf(stdout, "CL Error %d: %s\n", err, str); \
		exit(1); \
	}

#define TIMER_START clock_gettime(CLOCK_REALTIME, &timer1)
#define TIMER_END clock_gettime(CLOCK_REALTIME, &timer2)
#define MILLISECONDS (timer2.tv_sec - timer1.tv_sec) * 1000.0f + (timer2.tv_nsec - timer1.tv_nsec) / 1000000.0f
struct timespec timer1;
struct timespec timer2;

const char *KernelSourceFile = "VectorAdd.cl";
cl_platform_id platform_id;
cl_device_id device_id_gpu;
cl_device_id device_id_cpu;
cl_context context;
cl_command_queue commands_cpu;
cl_command_queue commands_gpu;
cl_program program;
cl_kernel kernel_compute;

const int warmup = 2;

enum scheme_t { CPU_ONLY, GPU_ONLY, CPU_GPU_STATIC, CPU_GPU_DYNAMIC };

enum scheme_t scheme = CPU_ONLY;
float ratio = 0.01;

cl_program createProgramFromSource(const char* filename, const cl_context context)
{
	FILE* kernelFile = NULL;
	kernelFile = fopen(filename, "r");
	if(!kernelFile)
		fprintf(stdout,"Error reading file.\n"), exit(0);
	fseek(kernelFile, 0, SEEK_END);
	size_t kernelLength = (size_t) ftell(kernelFile);
	char* kernelSource = (char *) calloc(1, sizeof(char)*kernelLength+1);
	rewind(kernelFile);
	if(fread((void *) kernelSource, kernelLength, 1, kernelFile) == 0) {
		fprintf(stderr, "Could not read source\n");
		exit(1);
	}
	kernelSource[kernelLength] = 0;
	fclose(kernelFile);
	
	// Create the compute program from the source buffer
	int err;
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &err);
	CHKERR(err, "Failed to create a compute program!");

	free(kernelSource);
	
	return program;
}

void setupGPU()
{
	// Retrieve an OpenCL platform
	cl_uint num_platforms = 0;
	int err = 0;
	err = clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id* platform_ids = (cl_platform_id*)(malloc(sizeof(cl_platform_id) * num_platforms));

	err = clGetPlatformIDs(num_platforms, platform_ids, NULL);
	CHKERR(err, "Failed to get a platform!");
	
	// Connect to a compute device
	int i = 0;
	for(i = 0; i < num_platforms; i++)
	{
		cl_device_id device_id;
		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
		if(err != CL_DEVICE_NOT_FOUND) 
		{
			CHKERR(err, "Failed to create a device group!");
			device_id_cpu = device_id;
		}
		
		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
		if(err != CL_DEVICE_NOT_FOUND) 
		{
			CHKERR(err, "Failed to create a device group!");
			device_id_gpu = device_id;	
		}
	}
	free(platform_ids);	


	// Create a compute context
	cl_device_id devices[2];
	devices[0] = device_id_gpu;
	devices[1] = device_id_cpu;
	context = clCreateContext(NULL, 2, devices, NULL, NULL, &err);
	CHKERR(err, "Failed to create a compute context!");

	// Create a command queue
	commands_cpu = clCreateCommandQueue(context, device_id_cpu, 0, &err);
	CHKERR(err, "Failed to create a command queue!");
	commands_gpu = clCreateCommandQueue(context, device_id_gpu, 0, &err);
	CHKERR(err, "Failed to create a command queue!");

	program = createProgramFromSource(KernelSourceFile, context);

	// Build the program executable
	err = clBuildProgram(program, 2, devices, NULL, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE)
	{
		char *log;
		size_t logLen;
		err = clGetProgramBuildInfo(program, device_id_gpu, CL_PROGRAM_BUILD_LOG, 0, NULL, &logLen);
		log = (char *) malloc(sizeof(char)*logLen);
		err = clGetProgramBuildInfo(program, device_id_gpu, CL_PROGRAM_BUILD_LOG, logLen, (void *) log, NULL);
		fprintf(stdout, "CL Error %d: Failed to build program! Log:\n%s", err, log);
		free(log);
		exit(1);
	}
	CHKERR(err, "Failed to build program!");

	// Create the compute kernel in the program we wish to run
	kernel_compute = clCreateKernel(program, "compute", &err);
	CHKERR(err, "Failed to create a compute kernel!");
}

long long t_length;
size_t t_offset;
pthread_mutex_t mutex;


struct dynamic_args
{
	cl_command_queue queue;
	cl_device_id dev_id;
	cl_kernel kernel;
	cl_mem* c;
};

void* dynamic_scheduler(void* args)
{
	struct dynamic_args* da = (struct dynamic_args*)args;
	cl_command_queue queue = da->queue;
	cl_device_id dev_id = da->dev_id;
	cl_kernel kernel = da->kernel;
	cl_mem* c = da->c;
	size_t local_size;
	cl_uint compute_units;
	clGetDeviceInfo(dev_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size, NULL);
	clGetDeviceInfo(dev_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);

	int err;
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), c);
	int work = compute_units * local_size;
	size_t offset = 0;
	while(1)
	{
		pthread_mutex_lock(&mutex);		
		if(t_length <= 0)
		{
			pthread_mutex_unlock(&mutex);
			return NULL;
		}
		if(t_length < work)
			work = t_length;
		offset = t_offset;
		size_t global_size = (work / local_size) * local_size + (work % local_size == 0 ? 0 : local_size);
		t_length -= global_size;
		t_offset += global_size;
		//printf("%d, %d\n", offset, t_length);
		if(global_size != 0)
		{
			err = clEnqueueNDRangeKernel(queue, kernel, 1, &offset, &global_size, &local_size, 0, NULL, NULL);
			CHKERR(err, "Errors setting kernel arguments1");
		}
		clFinish(queue);
		pthread_mutex_unlock(&mutex);	
	}
}

float runKernel(cl_mem a, cl_mem b, cl_mem* c_cpu, cl_mem* c_gpu, unsigned long length)
{
	int err;
	size_t local_size_cpu;
	size_t local_size_gpu;
	size_t global_size_cpu;
	size_t global_size_gpu;
	float executionTime;

	err = clSetKernelArg(kernel_compute, 0, sizeof(cl_mem), &a);
	err = clSetKernelArg(kernel_compute, 1, sizeof(cl_mem), &b);
	err = clSetKernelArg(kernel_compute, 3, sizeof(unsigned long), &length);
	CHKERR(err, "Errors setting kernel arguments");

	clGetDeviceInfo(device_id_cpu, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size_cpu, NULL);
	clGetDeviceInfo(device_id_gpu, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size_gpu, NULL);

	clFinish(commands_cpu);
	clFinish(commands_gpu);

	TIMER_START;
		if(scheme == CPU_ONLY)
		{
			err = clSetKernelArg(kernel_compute, 2, sizeof(cl_mem), c_cpu);
			global_size_cpu = (length / local_size_cpu) * local_size_cpu + (length % local_size_cpu == 0 ? 0 : local_size_cpu);
			err = clEnqueueNDRangeKernel(commands_cpu, kernel_compute, 1, NULL, &global_size_cpu, &local_size_cpu, 0, NULL, NULL);
			CHKERR(err, "Errors setting kernel arguments");
			clFinish(commands_cpu);
			clFinish(commands_gpu);
		}
		else if(scheme == GPU_ONLY)
		{
			err = clSetKernelArg(kernel_compute, 2, sizeof(cl_mem), c_gpu);
			global_size_gpu = (length / local_size_gpu) * local_size_gpu + (length % local_size_gpu == 0 ? 0 : local_size_gpu);
			err = clEnqueueNDRangeKernel(commands_gpu, kernel_compute, 1, NULL, &global_size_gpu, &local_size_gpu, 0, NULL, NULL);
			CHKERR(err, "Errors setting kernel arguments");
			clFinish(commands_cpu);
			clFinish(commands_gpu);
		}
		else if(scheme == CPU_GPU_STATIC)
		{
			err = clSetKernelArg(kernel_compute, 2, sizeof(cl_mem), c_gpu);
			size_t gpu_length = length * ratio;
			global_size_gpu = (gpu_length / local_size_gpu) * local_size_gpu + (gpu_length % local_size_gpu == 0 ? 0 : local_size_gpu);
			err = clEnqueueNDRangeKernel(commands_gpu, kernel_compute, 1, NULL, &global_size_gpu, &local_size_gpu, 0, NULL, NULL);
			CHKERR(err, "Errors setting kernel arguments2");
			
			err = clSetKernelArg(kernel_compute, 2, sizeof(cl_mem), c_cpu);
			size_t cpu_length = length - global_size_gpu;
			global_size_cpu = (cpu_length / local_size_cpu) * local_size_cpu + (cpu_length % local_size_cpu == 0 ? 0 : local_size_cpu);
			if(global_size_cpu != 0)
			{
				err = clEnqueueNDRangeKernel(commands_cpu, kernel_compute, 1, &global_size_gpu, &global_size_cpu, &local_size_cpu, 0, NULL, NULL);
				CHKERR(err, "Errors setting kernel arguments1");
			}
			clFinish(commands_cpu);
			clFinish(commands_gpu);
		}
		else if(scheme == CPU_GPU_DYNAMIC)
		{
			pthread_t threads[2];

			struct dynamic_args cpu;
			struct dynamic_args gpu;
			int rc;
			
			pthread_mutex_init(&mutex, NULL);
			t_length = length;
			t_offset = 0;
	
			cpu.queue = commands_cpu;
			cpu.dev_id = device_id_cpu;
			cpu.kernel = kernel_compute;		
			cpu.c =  c_cpu;
			rc = pthread_create(&threads[0], NULL, dynamic_scheduler, (void*)&cpu);
			
			gpu.queue = commands_gpu;
			gpu.dev_id = device_id_gpu;
			gpu.kernel = kernel_compute;
			gpu.c = c_gpu;
			rc = pthread_create(&threads[1], NULL, dynamic_scheduler, (void*)&gpu);
			void* status;
			rc = pthread_join(threads[0], &status); 
			rc = pthread_join(threads[1], &status); 
		}
	TIMER_END;
	executionTime = MILLISECONDS;

	return executionTime;
}

void vadd_default(unsigned char* a, unsigned char* b, unsigned char* c, unsigned long length, float* data_time, float* kernel_time)
{
	int err;
	cl_mem dev_a;
	cl_mem dev_b;
	cl_mem dev_c_cpu;
	cl_mem dev_c_gpu;

	unsigned char* temp = calloc(1, sizeof(*temp) * length);

	dev_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*a) * length, NULL, &err);
	dev_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*b) * length, NULL, &err);
	dev_c_cpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY,sizeof(*c) * length, NULL, &err);
	dev_c_gpu = clCreateBuffer(context, CL_MEM_WRITE_ONLY,sizeof(*c) * length, NULL, &err);
	CHKERR(err, "Errors creating buffers");

	clFinish(commands_cpu);
	TIMER_START;
	err = clEnqueueWriteBuffer(commands_cpu, dev_a, CL_TRUE, 0, sizeof(*a) * length, a, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands_cpu, dev_b, CL_TRUE, 0, sizeof(*b) * length, b, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands_cpu, dev_c_gpu, CL_TRUE, 0, sizeof(*c) * length, temp, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands_cpu, dev_c_cpu, CL_TRUE, 0, sizeof(*c) * length, temp, 0, NULL, NULL);
	clFinish(commands_cpu);
	TIMER_END;
	*data_time += MILLISECONDS;
	CHKERR(err, "Errors writing buffers");

	*kernel_time += runKernel(dev_a, dev_b, &dev_c_cpu, &dev_c_gpu, length);
	
	clFinish(commands_cpu);
	TIMER_START;
	err = clEnqueueReadBuffer(commands_cpu, dev_c_cpu, CL_TRUE, 0, sizeof(*c) * length, c, 0, NULL, NULL);
	err = clEnqueueReadBuffer(commands_gpu, dev_c_gpu, CL_TRUE, 0, sizeof(*c) * length, temp, 0, NULL, NULL);
	int i;
	for(i = 0; i < length; i++)
	{
		if(temp[i] != 0)
			c[i] = temp[i];
	}
	clFinish(commands_cpu);
	TIMER_END;
	*data_time += MILLISECONDS;
	CHKERR(err, "Errors reading buffers");

	free (temp);
	
	clReleaseMemObject(dev_a);
	clReleaseMemObject(dev_b);
	clReleaseMemObject(dev_c_cpu);
	clReleaseMemObject(dev_c_gpu);
}

void fillArray(unsigned char* nums, unsigned long length)
{
	int i;
	for(i = 0; i < length; i++)
	{
		nums[i] = rand() % 256;
	}
}

void serial_vector_add(unsigned char* a, unsigned char* b, unsigned char* c, const unsigned int len)
{
	int i;
	for(i = 0; i < len; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void verify_answer(unsigned char* toCheck, unsigned char* answer, const unsigned int len)
{
	int i;
	for(i = 0; i < len; i++)
	{
		if(toCheck[i] != answer[i])
			fprintf(stderr,"Answers differ at position %d (%d, %d)\n", i, toCheck[i], answer[i]);
	}
}

int main(int argc, char** argv)
{
	unsigned char* nums_1;
	unsigned char* nums_2;
	unsigned char* nums_3;
	unsigned char* nums_check;
	unsigned char* scheme_name;
	unsigned long length = atoi(argv[1]);
	unsigned int iters = atoi(argv[2]);
	switch(atoi(argv[3]))
	{
		case 0: scheme = CPU_ONLY;
			scheme_name = (unsigned char*)"c";
			break;
		case 1: scheme = GPU_ONLY;
			scheme_name = (unsigned char*)"g";
			break;
		case 2: scheme = CPU_GPU_STATIC;
			scheme_name = (unsigned char*)"cg-s";
			if(argc > 4)
				ratio = atof(argv[4]);
			break;
		case 3: scheme = CPU_GPU_DYNAMIC;
			scheme_name = (unsigned char*)"cg-d";
			break;
		default:
			fprintf(stderr, "Error: no scheme specified\n");
			exit(1);
	}
	nums_1 = malloc(sizeof(*nums_1) *  length);
	nums_2 = malloc(sizeof(*nums_2) *  length);
	nums_3 = malloc(sizeof(*nums_3) *  length);
	nums_check = malloc(sizeof(*nums_check) *  length);

	setupGPU();	

	srand(time(0));
	fillArray(nums_1, length);
	fillArray(nums_2, length);
	serial_vector_add(nums_1, nums_2, nums_check, length);

	float data_time = 0;
	float exec_time = 0;
	
//	fprintf(stdout, "Trial no.\tData Access Scheme\tData Elements\tData Transfer Time\tKernel Execution Time\n");

	int i;
	//fprintf(stdout,"Default Flags (Read/Write Buffers)\n");
	for(i = 0; i < iters+warmup; i++)
	{
		memset(nums_3, 0, sizeof(unsigned char) * length);
		vadd_default(nums_1, nums_2, nums_3, length, &data_time, &exec_time);
		verify_answer(nums_3, nums_check, length);	
		if(i >= warmup)
		{
			fprintf(stdout,"%d\tVectorAdd\t%s\t%f\t%lu\t%f\t%f\n", i - warmup, scheme_name, ratio, length, data_time, exec_time);
		}
		data_time = 0;
		exec_time = 0;
	}

	fflush(stdout);
	free(nums_1);
	free(nums_2);
	free(nums_3);
	return 0;
}