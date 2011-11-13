__kernel void compute(__global unsigned char* a,
			__global unsigned char* b,
			__global unsigned char* c,
			const unsigned long length)
{
	unsigned int tid = get_global_id(0);
	if(tid < length)
	{
		c[tid] = a[tid] + b[tid];
	}
}
