__kernel void compute(__global unsigned long* buffer,
			__global unsigned long* reduction,
			const unsigned long length,
			const unsigned long chunk)
{
	unsigned int tid = get_global_id(0);
	int start = tid * chunk;
	int end = start + chunk;
	int sum = 0;
	if(end > length)
		end = length;
	for(int i = start; i < end; i++)
		sum += buffer[i];
	reduction[tid] = sum;
} 
