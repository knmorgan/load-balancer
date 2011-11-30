__kernel void compute(__global unsigned long* buffer,
			const unsigned long length)
{
	unsigned int tid = get_global_id(0);
	if(tid < length)
	{
		unsigned long i, sum = 0;
		for(i = 0; i < length; ++i) {
			sum += buffer[i];
		}
	}
}
