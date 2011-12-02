__kernel void compute(__global unsigned long* buffer,
			__global unsigned long* reduction,
			const unsigned long length,
			const unsigned long chunk,
			__local unsigned long* local_mem)	
{
	unsigned int tid = get_global_id(0);
	unsigned int lid = get_local_id(0);
	local_mem[lid] = 0;
	int start = tid * chunk;
	int end = start + chunk;
	int sum = 0;
	if(end > length)
		end = length;
	for(int i = start; i < end; i++)
		sum += buffer[i];
	local_mem[lid] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
	unsigned int size = get_local_size(0) / 2;
	while(size > 0)
	{
		if(lid < size)
			local_mem[lid] += local_mem[lid + size];
		size = size / 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0)
		reduction[get_group_id(0)] = local_mem[0];
} 
