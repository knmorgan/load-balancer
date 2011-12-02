__kernel void compute(__global unsigned long* buffer,
			__global unsigned long* reduction,
			const unsigned long length,
			const unsigned long chunk,
			__local unsigned long* local_mem)	
{
	unsigned int tid = get_local_size(0) * get_group_id(0) + get_local_id(0);
	unsigned int lid = get_local_id(0);
	local_mem[lid] = 0;
	if(tid < length)
		local_mem[lid] = buffer[tid];
	local_mem[get_local_size(0) + lid] = buffer[tid + get_local_size(0)];

	barrier(CLK_LOCAL_MEM_FENCE);
	unsigned int size = get_local_size(0);
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
