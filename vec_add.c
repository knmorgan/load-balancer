__kernel void vec_add(__global float* a, __global float* b, __global float* c) {
	int gid = get_global_id(0);
	c[gid] = a[gid] + b[gid];
}
