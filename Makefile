OPENCL_LIB_DIR = /home/zjing/AMD-APP-SDK-v2.4-lnx64/lib/x86_64/
OPENCL_INCLUDE_DIR = /home/zjing/AMD-APP-SDK-v2.4-lnx64/include
LD_FLAGS = -O3


all: VectorAdd.c
	gcc -I $(OPENCL_INCLUDE_DIR) -L $(OPENCL_LIB_DIR) VectorAdd.c -o VectorAdd -lOpenCL -lrt
