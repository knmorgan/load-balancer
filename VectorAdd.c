#include <stdio.h>
#include <string.h>
#include <time.h>

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

enum scheme_t scheme = GPU_ONLY;

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
	fread((void *) kernelSource, kernelLength, 1, kernelFile);
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
	int num_platforms = 0;
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

float runKernel(cl_mem a, cl_mem b, cl_mem* c, unsigned long length)
{
	int err;
	size_t local_size_cpu;
	size_t local_size_gpu;
	size_t global_size_cpu;
	size_t global_size_gpu;
	float executionTime;

	err = clSetKernelArg(kernel_compute, 0, sizeof(cl_mem), &a);
	err = clSetKernelArg(kernel_compute, 1, sizeof(cl_mem), &b);
	err = clSetKernelArg(kernel_compute, 2, sizeof(cl_mem), c);
	err = clSetKernelArg(kernel_compute, 3, sizeof(unsigned long), &length);
	CHKERR(err, "Errors setting kernel arguments");

	clGetDeviceInfo(device_id_cpu, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size_cpu, NULL);
	clGetDeviceInfo(device_id_gpu, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size_gpu, NULL);

	clFinish(commands_cpu);
	clFinish(commands_gpu);

	TIMER_START;
		if(scheme == CPU_ONLY)
		{
			global_size_cpu = (length / local_size_cpu) * local_size_cpu + (length % local_size_cpu == 0 ? 0 : local_size_cpu);
			err = clEnqueueNDRangeKernel(commands_cpu, kernel_compute, 1, NULL, &global_size_cpu, &local_size_cpu, 0, NULL, NULL);
			CHKERR(err, "Errors setting kernel arguments");
			clFinish(commands_cpu);
			clFinish(commands_gpu);
		}
		if(scheme == GPU_ONLY)
		{
			global_size_gpu = (length / local_size_gpu) * local_size_gpu + (length % local_size_gpu == 0 ? 0 : local_size_gpu);
			err = clEnqueueNDRangeKernel(commands_gpu, kernel_compute, 1, NULL, &global_size_gpu, &local_size_gpu, 0, NULL, NULL);
			CHKERR(err, "Errors setting kernel arguments");
			clFinish(commands_cpu);
			clFinish(commands_gpu);
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
	cl_mem dev_c;

	dev_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*a) * length, NULL, &err);
	dev_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*b) * length, NULL, &err);
	dev_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,sizeof(*c) * length, NULL, &err);
	CHKERR(err, "Errors creating buffers");

	clFinish(commands_cpu);
	TIMER_START;
	err = clEnqueueWriteBuffer(commands_cpu, dev_a, CL_TRUE, 0, sizeof(*a) * length, a, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands_cpu, dev_b, CL_TRUE, 0, sizeof(*b) * length, b, 0, NULL, NULL);
	clFinish(commands_cpu);
	TIMER_END;
	*data_time += MILLISECONDS;
	CHKERR(err, "Errors writing buffers");

	*kernel_time += runKernel(dev_a, dev_b, &dev_c, length);
	
	clFinish(commands_cpu);
	TIMER_START;
	err = clEnqueueReadBuffer(commands_cpu, dev_c, CL_TRUE, 0, sizeof(*c) * length, c, 0, NULL, NULL);
	clFinish(commands_cpu);
	TIMER_END;
	*data_time += MILLISECONDS;
	CHKERR(err, "Errors reading buffers");

	
	clReleaseMemObject(dev_a);
	clReleaseMemObject(dev_b);
	clReleaseMemObject(dev_c);
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
	unsigned long length = atoi(argv[1]);
	unsigned int iters = atoi(argv[2]);
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
			fprintf(stdout,"%d\tVectorAdd\tDefault\t%lu\t%f\t%f\n", i - warmup, length, data_time, exec_time);
		}
		data_time = 0;
		exec_time = 0;
	}

	fflush(stdout);
	free(nums_1);
	free(nums_2);
	free(nums_3);
}
