#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

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

#define TOTAL_TIMER_START clock_gettime(CLOCK_REALTIME, &total_timer1)
#define TOTAL_TIMER_END clock_gettime(CLOCK_REALTIME, &total_timer2)
#define TOTAL_MILLISECONDS (total_timer2.tv_sec - total_timer1.tv_sec) * 1000.0f + (total_timer2.tv_nsec - total_timer1.tv_nsec) / 1000000.0f
struct timespec total_timer1;
struct timespec total_timer2;

typedef unsigned long reduce_t;

//OpenCL Constructs
const char *KernelSourceFile_cpu = "Reduction_CPU.cl";
const char *KernelSourceFile_gpu = "Reduction_GPU.cl";
cl_platform_id platform_id;
cl_device_id device_id_gpu;
cl_device_id device_id_cpu;
cl_context context_cpu;
cl_context context_gpu;
cl_command_queue commands_cpu;
cl_command_queue commands_gpu;
cl_program program;
cl_kernel kernel_compute_cpu;
cl_kernel kernel_compute_gpu;

cl_event event_gpu;
cl_event event_cpu;

//Number of iterations to warmup caches
const int warmup = 2;

enum scheme_t { CPU_ONLY, GPU_ONLY, CPU_GPU_STATIC, CPU_GPU_DYNAMIC };
enum scheme_t scheme = CPU_ONLY;
float ratio = 0.01;

//Data
unsigned long length;
reduce_t* h_a;
reduce_t* h_b;
reduce_t h_check;
cl_mem dc_a;
cl_mem dc_b;
cl_mem dg_a;
cl_mem dg_b;
reduce_t ans_gpu = 0;
reduce_t ans_cpu = 0;
reduce_t ans;

// Struct for passing arguments to dynamic_scheduler
struct dynamic_args
{
	int isGPU;
	float data_time;
	float exec_time;
};


//Function Prototypes
void fillArray(reduce_t* nums, const unsigned long length);
void verify_answer(reduce_t* toCheck, reduce_t* answer, const unsigned int len);
void serial_reduce(reduce_t* a, reduce_t* c, const unsigned int len);

void test_setup();
void test_init();

void test_chunk_setup(cl_context context, cl_command_queue commands, size_t global_size, size_t offset, int isGPU);
void test_chunk_kernel(cl_context context, cl_command_queue commands, cl_device_id device, cl_kernel kernel, size_t global_size, size_t offset, int isGPU);
void test_chunk_cleanup(cl_context context, cl_command_queue commands, size_t global_size, size_t offset, int isGPU);

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

cl_kernel create_kernel(const char* filename, const char* kernel, const cl_context context, const cl_device_id device)
{
	cl_kernel kernel_compute;
	// Create a command queue
	cl_program program = createProgramFromSource(filename, context);

	// Build the program executable
	//int err = clBuildProgram(program, 1, &device, "-cl-opt-disable", NULL, NULL);
	int err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
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
	kernel_compute = clCreateKernel(program, kernel, &err);
	CHKERR(err, "Failed to create a compute kernel!");
	
	return kernel_compute;
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


	if(scheme != GPU_ONLY)
	{
		context_cpu = clCreateContext(NULL, 1, &device_id_cpu, NULL, NULL, &err);
		CHKERR(err, "Failed to create a compute context!");
		commands_cpu = clCreateCommandQueue(context_cpu, device_id_cpu, 0, &err);
		CHKERR(err, "Failed to create a command queue!");
		kernel_compute_cpu = create_kernel(KernelSourceFile_cpu, "compute", context_cpu, device_id_cpu);
	}

	if(scheme != CPU_ONLY)
	{
		context_gpu = clCreateContext(NULL, 1, &device_id_gpu, NULL, NULL, &err);
		CHKERR(err, "Failed to create a compute context!");
		commands_gpu = clCreateCommandQueue(context_gpu, device_id_gpu, 0, &err);
		CHKERR(err, "Failed to create a command queue!");
		kernel_compute_gpu = create_kernel(KernelSourceFile_gpu, "compute", context_gpu, device_id_gpu);
	}

}

long long t_length;
size_t t_offset;
pthread_mutex_t mutex;


void* dynamic_scheduler(void* argv)
{
	struct dynamic_args* args = argv;
	int isGPU = args->isGPU;
	size_t local_size;
	struct timespec time_start, time_end;
	
	cl_device_id device;
	cl_context context;
	cl_kernel kernel;
	cl_command_queue commands;	

	device = isGPU ? device_id_gpu : device_id_cpu;
	context = isGPU ? context_gpu : context_cpu;
	kernel = isGPU ? kernel_compute_gpu : kernel_compute_cpu;
	commands = isGPU ? commands_gpu : commands_cpu;

	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size, NULL);

	size_t offset = 0;
	size_t global_size = 1024 * 80;
	while(1)
	{
		pthread_mutex_lock(&mutex);		
		if(t_length <= 0)
		{
			pthread_mutex_unlock(&mutex);
			return NULL;
		}
		offset = t_offset;
		t_length -= global_size;
		t_offset += global_size;
		pthread_mutex_unlock(&mutex);
	
		global_size = global_size + offset > length ? length - offset : global_size;


		clock_gettime(CLOCK_REALTIME, &time_start);
		test_chunk_setup(context, commands, global_size, offset, isGPU);
		clFinish(commands);
		clock_gettime(CLOCK_REALTIME, &time_end);
		args->data_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0f + (time_end.tv_nsec - time_start.tv_nsec) / 1000000.0f;

		clock_gettime(CLOCK_REALTIME, &time_start);
		test_chunk_kernel(context, commands, device, kernel, global_size, offset, isGPU);
		clFinish(commands);
		clock_gettime(CLOCK_REALTIME, &time_end);
		args->exec_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0f + (time_end.tv_nsec - time_start.tv_nsec) / 1000000.0f;

		clock_gettime(CLOCK_REALTIME, &time_start);
		test_chunk_cleanup(context, commands, global_size, offset, isGPU);
		clFinish(commands);
		clock_gettime(CLOCK_REALTIME, &time_end);
		args->data_time += (time_end.tv_sec - time_start.tv_sec) * 1000.0f + (time_end.tv_nsec - time_start.tv_nsec) / 1000000.0f;
	}
}

void test_setup()
{
	ans_gpu = 0;
	ans_cpu = 0;
	fillArray(h_a, length);
	serial_reduce(h_a, &h_check, length);
}

void test_init()
{
}

void test_chunk_setup(cl_context context, cl_command_queue queue, size_t size, size_t offset, int isGPU)
{
	if(size == 0)
		return;
	cl_mem* d_a = isGPU ? &dg_a : &dc_a;
	cl_mem* d_b = isGPU ? &dg_b : &dc_b;
	cl_mem_flags a_flags = CL_MEM_READ_WRITE;
	cl_mem_flags b_flags = CL_MEM_READ_WRITE;
	void* a_mem = NULL;
	void* b_mem = NULL;

	if(!isGPU)
	{
		a_flags |= CL_MEM_USE_HOST_PTR;
		a_mem = h_a;
	}

	int err;
	*d_a = clCreateBuffer(context, a_flags, sizeof(*h_a) * size, a_mem, &err);
	CHKERR(err, "Failed to create chunk buffers!");
	*d_b = clCreateBuffer(context, b_flags, sizeof(*h_b) * size, b_mem, &err);
	CHKERR(err, "Failed to create chunk buffers!");

	err = clEnqueueWriteBuffer(queue, *d_a, CL_FALSE, 0, sizeof(*h_a) * size, h_a + offset, 0, NULL, NULL);
	CHKERR(err, "Failed to write chunk buffer A!");
}

void test_chunk_kernel(cl_context context, cl_command_queue queue, cl_device_id device, cl_kernel kernel, size_t size, size_t offset, int isGPU)
{
	if(size == 0)
		return;
	cl_mem* d_a = isGPU ? &dg_a : &dc_a;
	cl_mem* d_b = isGPU ? &dg_b : &dc_b;
	size_t chunk = isGPU ? 2 : size;

	cl_event* event = isGPU ? &event_gpu : &event_cpu;

	size_t local_size;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_size, NULL);
	if(!isGPU)
		local_size = 1;

	int err = clSetKernelArg(kernel, 0, sizeof(cl_mem), d_a);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), d_b);
	err |= clSetKernelArg(kernel, 2, sizeof(size_t), &size);
	err |= clSetKernelArg(kernel, 3, sizeof(size_t), &chunk);
	if(isGPU)
		err |= clSetKernelArg(kernel, 4, sizeof(reduce_t) * local_size * 2, NULL);
	CHKERR(err, "Errors setting kernel arguments");

	size_t groups = size / local_size / chunk + (size % (local_size*chunk) == 0 ? 0 : 1);
	size_t global_size = groups * local_size;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, event);
	CHKERR(err, "Failed to run kernel!");
	if(groups != 1)
	{
		cl_mem temp = *d_a;
		*d_a = *d_b;
		*d_b = temp;
		test_chunk_kernel(context, queue, device, kernel, groups, offset, isGPU);
	}
}

void test_chunk_cleanup(cl_context context, cl_command_queue queue, size_t size, size_t offset, int isGPU)
{
	if(size == 0)
		return;
	cl_mem* d_a = isGPU ? &dg_a : &dc_a;
	cl_mem* d_b = isGPU ? &dg_b : &dc_b;

	reduce_t* ans = isGPU ? &ans_gpu : &ans_cpu;

	reduce_t answer;
	int err = clEnqueueReadBuffer(queue, *d_b, CL_TRUE, 0, sizeof(reduce_t), &answer, 0, NULL, NULL);
	CHKERR(err, "Failed to read back buffer!");

	*ans += answer;
	clReleaseMemObject(*d_a);
	clReleaseMemObject(*d_b);
}

void test_cleanup()
{
	ans = ans_cpu + ans_gpu;
	//verify_answer(h_c, h_check, length);
}

void run_test(float* data_time, float* exec_time, float* total_time)
{
	test_setup();
	TOTAL_TIMER_START;
	TIMER_START;
	test_init();	
	TIMER_END;
	*data_time += MILLISECONDS;
	if(scheme == GPU_ONLY)
	{
		TIMER_START;
		test_chunk_setup(context_gpu, commands_gpu, length, 0, 1);
		clFinish(commands_gpu);
		TIMER_END;
		*data_time += MILLISECONDS;
		
		TIMER_START;
		test_chunk_kernel(context_gpu, commands_gpu, device_id_gpu, kernel_compute_gpu, length, 0, 1);
		clFinish(commands_gpu);
		TIMER_END;
		*exec_time += MILLISECONDS;
		
		TIMER_START;
		test_chunk_cleanup(context_gpu, commands_gpu, length, 0, 1);
		clFinish(commands_gpu);
		TIMER_END;
		*data_time += MILLISECONDS;
	}
	else if(scheme == CPU_ONLY)
	{
		TIMER_START;
		test_chunk_setup(context_cpu, commands_cpu, length, 0, 0);
		clFinish(commands_cpu);
		TIMER_END;
		*data_time += MILLISECONDS;
		
		TIMER_START;
		test_chunk_kernel(context_cpu, commands_cpu, device_id_cpu, kernel_compute_cpu, length, 0, 0);
		clFinish(commands_cpu);
		TIMER_END;
		*exec_time += MILLISECONDS;
		
		TIMER_START;
		test_chunk_cleanup(context_cpu, commands_cpu, length, 0, 0);
		clFinish(commands_cpu);
		TIMER_END;
		*data_time += MILLISECONDS;
	}
	else if(scheme == CPU_GPU_STATIC)
	{
		size_t gpu_size = length * ratio;
		TIMER_START;
		test_chunk_setup(context_cpu, commands_cpu, length - gpu_size, 0, 0);
		test_chunk_setup(context_gpu, commands_gpu, gpu_size, length - gpu_size, 1);
		clFinish(commands_cpu);
		clFinish(commands_gpu);
		TIMER_END;
		*data_time += MILLISECONDS;
	
		TIMER_START;
		test_chunk_kernel(context_gpu, commands_gpu, device_id_gpu, kernel_compute_gpu, gpu_size, length - gpu_size, 1);
		test_chunk_kernel(context_cpu, commands_cpu, device_id_cpu, kernel_compute_cpu, length - gpu_size, 0, 0);
		clFlush(commands_gpu);
		clFlush(commands_cpu);
		//cl_event events[2] = {event_cpu, event_gpu};
		//clEnqueueWaitForEvents(commands_cpu, 2, events);
		clFinish(commands_cpu);
		clFinish(commands_gpu);
		TIMER_END;
		*exec_time += MILLISECONDS;
		
		TIMER_START;
		test_chunk_cleanup(context_cpu, commands_cpu, length - gpu_size, 0, 0);
		test_chunk_cleanup(context_gpu, commands_gpu, gpu_size, length - gpu_size, 1);
		clFinish(commands_cpu);
		clFinish(commands_gpu);
		TIMER_END;
		*data_time += MILLISECONDS;
	}
	else if(scheme == CPU_GPU_DYNAMIC)
	{
		pthread_t threads[2];
		int rc;
		void* status;
		struct dynamic_args cpu_args = {0, 0, 0};
		struct dynamic_args gpu_args = {1, 0, 0};
		
		pthread_mutex_init(&mutex, NULL);
		t_length = length;
		t_offset = 0;
		
		TIMER_START;
		rc = pthread_create(&threads[0], NULL, dynamic_scheduler, &gpu_args);
		rc = pthread_create(&threads[1], NULL, dynamic_scheduler, &cpu_args);
		rc = pthread_join(threads[0], &status); 
		rc = pthread_join(threads[1], &status); 
		TIMER_END;

		*data_time = MILLISECONDS;//cpu_args.data_time + gpu_args.data_time;
		//*exec_time = cpu_args.exec_time + gpu_args.exec_time;
	}
	else
	{
		fprintf(stderr, "Scheme not supported.\n");
		abort();
	}
	test_cleanup();	
	TOTAL_TIMER_END;
	*total_time = TOTAL_MILLISECONDS;
}

void fillArray(reduce_t* nums, unsigned long length)
{
	int i;
	for(i = 0; i < length; i++)
	{
		nums[i] = rand() % 256;
	}
}

void serial_reduce(reduce_t* a, reduce_t* check, const unsigned int len)
{
	int i;
	int sum = 0;
	for(i = 0; i < len; i++)
	{
		sum += a[i];
	}
	*check = sum;
}

void verify_answer(reduce_t* toCheck, reduce_t* answer, const unsigned int len)
{
	if(*toCheck != *answer)
		fprintf(stderr,"Answers differ at position (%lu, %lu)\n", *toCheck, *answer);
}

int main(int argc, char** argv)
{
	const char* scheme_name;

	length = atoi(argv[1]);
	unsigned int iters = atoi(argv[2]);
	switch(atoi(argv[3]))
	{
		case 0: scheme = CPU_ONLY;
			scheme_name = "c";
			break;
		case 1: scheme = GPU_ONLY;
			scheme_name = "g";
			break;
		case 2: scheme = CPU_GPU_STATIC;
			scheme_name = "cg-s";
			if(argc > 4)
				ratio = atof(argv[4]);
			break;
		case 3: scheme = CPU_GPU_DYNAMIC;
			scheme_name = "cg-d";
			break;
		default:
			fprintf(stderr, "Error: no scheme specified\n");
			exit(1);
	}
	h_a = malloc(sizeof(*h_a) *  length);

	setupGPU();	

	srand(time(0));

	float data_time = 0;
	float exec_time = 0;
	float total_time = 0;
	
	int i;
	for(i = 0; i < iters+warmup; i++)
	{
		run_test(&data_time, &exec_time, &total_time);
		if(i >= warmup)
		{
			fprintf(stdout,"%d\tVectorAdd\t%s\t%f\t%lu\t%f\t%f\t%f\n", i - warmup, scheme_name, ratio, length, data_time, exec_time, total_time);
		}
		data_time = 0;
		exec_time = 0;
	}

	fflush(stdout);
	free(h_a);
	return 0;
}
