__kernel void addVectors(__global const char *a, 
	__global const char *b,
	__global char *tc) {
		
		__global const float2 *tmpa = (__global const float2*)(a + 64 * 4);
		__global const float2 *tmpb = (__global const float2*)(b + 64 * 4);
		__global float2 *c = (__global float2*)tc;
		int gid = get_global_id(0);
		__local float t[2 * 64];
		int lid = get_local_id(0);
		t[0 + lid*2] = tmpa[gid].x + tmpb[gid].x;
		t[1 + lid*2] = tmpa[gid].y + tmpb[gid].y;
		float tt[2];
		tt[0] = t[lid*2+0];
		tt[1] = t[lid*2+1];
//		c[gid + 32] = *(float2*)&tt;
		c[gid + 32] = *(__global float2*)&t[lid*2];
	}
