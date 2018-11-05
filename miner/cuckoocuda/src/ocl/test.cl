__kernel void test(__global ulong4 * buffer){
	int id = get_global_id(0);
	//buffer[id] += id;
	uint2 v2 = (uint2)(0,1);
	ulong lv = ((ulong)v2.y << 32 | (ulong)v2.x);
	printf("%lu\n", lv);
	printf("%lu %lu %lu %lu\n", buffer[0].x, buffer[0].y, buffer[0].z, buffer[0].w);
}
