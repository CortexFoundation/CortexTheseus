typedef uint8 u8;
typedef uint16 u16;
typedef uint u32;
typedef ulong u64;

typedef u32 node_t;
typedef u64 nonce_t;

#define DUCK_SIZE_A 129

#define DUCK_A_EDGES (DUCK_SIZE_A * 1024)
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64)

#define EDGEBITS 29
// number of edges
#define NEDGES ((node_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK (NEDGES - 1)

#define CTHREADS 1024
#define BKTMASK4K (4096-1)
#define BKTMASK16K (16384-1)

#define BKTGRAN 32

inline unsigned int ld_cs_u32(__global const unsigned int *p_src)
{
#ifdef NVIDIA
	unsigned int n_result;
	asm("ld.global.cs.u32 %0, [%1];" : "=r"(n_result) : "l"(p_src));
	return n_result;
#else // NVIDIA
	return *p_src; // generic
#endif // NVIDIA
}

inline unsigned int ld_nc_u32(__global const unsigned int *p_src)
{
#ifdef NVIDIA
	unsigned int n_result;
	asm("ld.global.nc.u32 %0, [%1];" : "=r"(n_result) : "l"(p_src));
	return n_result;
#else // NVIDIA
	return *p_src; // generic
#endif // NVIDIA
}

inline uint2 ld_nc_u32_v2(__global const uint2 *p_src)
{
#ifdef NVIDIA
	uint2 n_result;
	asm("ld.global.nc.v2.u32 {%0,%1}, [%2];"  : "=r"(n_result.x), "=r"(n_result.y) : "l"(p_src));
	return n_result;
#else // NVIDIA
	return *p_src; // generic
#endif // NVIDIA
}

inline void st_cg_u32(__global  uint *p_dest, const uint n_value)
{
#ifdef NVIDIA
	asm("st.global.cg.u32 [%0], %1;" : : "l"(p_dest), "r"(n_value));
#else // NVIDIA
	*p_dest = n_value; // generic
#endif // NVIDIA
}

inline void st_cg_u32_v2(__global  uint2 *p_dest, const uint2 n_value)
{
#ifdef NVIDIA
	asm("st.global.cg.v2.u32 [%0], {%1, %2};" :: "l"(p_dest), "r"(n_value.x), "r"(n_value.y));
#else // NVIDIA
	*p_dest = n_value; // generic
#endif // NVIDIA
}

inline void st_cs_u32_v4(__global  uint4 *p_dest, const uint4 n_value)
{
#ifdef NVIDIA
	asm("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(p_dest), "r"(n_value.x), "r"(n_value.y), "r"(n_value.z), "r"(n_value.w));
#else // NVIDIA
	*p_dest = n_value; // generic
#endif // NVIDIA
}

#define SIPROUND \
  do { \
    v0 += v1; v2 += v3; v1 = rotate(v1,(ulong)13); \
    v3 = rotate(v3,(ulong)16); v1 ^= v0; v3 ^= v2; \
    v0 = rotate(v0,(ulong)32); v2 += v1; v0 += v3; \
    v1 = rotate(v1,(ulong)17);   v3 = rotate(v3,(ulong)21); \
    v1 ^= v2; v3 ^= v0; v2 = rotate(v2,(ulong)32); \
  } while(0)

uint _dipnode(ulong v0i, ulong v1i, ulong v2i, ulong v3i, uint nce, uint uorv) {
	ulong nonce = 2 * nce + uorv;
	ulong v0 = v0i, v1 = v1i, v2 = v2i, v3 = v3i ^ nonce;
	ulong x = (v0 * v1 * v2  * v3);
	x ^= (x) >> 12;
	x ^= (x) << 25;
	x ^= (x) >> 27;
	return (x * 2685821657736338717ul) & EDGEMASK;
}

uint2 _dipnode2(ulong v0i, ulong v1i, ulong v2i, ulong v3i, uint nce) {
	ulong2 nonce = (ulong2)(2 * nce + 0, 2 * nce + 1);
	ulong2 v0 = v0i, v1 = v1i, v2 = v2i, v3 = v3i ^ nonce;
	ulong2 x = (v0 * v1 * v2  * v3);
	x ^= (x) >> 12;
	x ^= (x) << 25;
	x ^= (x) >> 27;
	return convert_uint2((x * 2685821657736338717ul) & EDGEMASK);
}

uint dipnode(ulong v0i, ulong v1i, ulong v2i, ulong v3i, uint nce, uint uorv) {
	ulong nonce = 2 * nce + uorv;
	ulong v0 = v0i, v1 = v1i, v2 = v2i, v3 = v3i ^ nonce;
	SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
	return (v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}

uint2 dipnode2(ulong v0i, ulong v1i, ulong v2i, ulong v3i, uint nce) {
	ulong2 nonce = (ulong2)(2 * nce + 0, 2 * nce + 1);
	ulong2 v0 = v0i, v1 = v1i, v2 = v2i, v3 = v3i ^ nonce;
	SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
	return convert_uint2((v0 ^ v1 ^ v2  ^ v3) & EDGEMASK);
}

void Increase2bCounter(__local u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	u32 old = atomic_or(ecounters + word, mask) & mask;

	if (old > 0)
		atomic_or(ecounters + word + 4096, mask);
}

bool Read2bCounter(__local u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	return (ecounters[word + 4096] & mask) > 0;
}


__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel  void FluffySeed1A(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, __global uint8 * buffer, __global int * indexes)
{
	const int gid = get_global_id(0);
	const short lid = get_local_id(0);

	__local uint tmp[64][15];
	__local int counters[64];

	counters[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (short i = 0; i < 1024 * 4; i++)
	{
		u64 nonce = gid * (1024 * 4) + i;

		uint hash = dipnode(v0i, v1i, v2i, v3i, nonce, 0);

		int bucket = hash & (63);

		barrier(CLK_LOCAL_MEM_FENCE);

		int counter = min((int)atomic_inc(counters + bucket), (int)14);

		tmp[bucket][counter] = nonce;

		barrier(CLK_LOCAL_MEM_FENCE);

		{
			int localIdx = min(15, counters[lid]);

			if (localIdx >= 8)
			{
				int newCount = (localIdx - 8);
				counters[lid] = newCount;

				int cnt = min((int)atomic_add(indexes + lid, 8), (int)(DUCK_A_EDGES_64 - 8));

				int idx = (lid * DUCK_A_EDGES_64 + cnt) / 8;

				buffer[idx] = (uint8)
					(
						tmp[lid][0],
						tmp[lid][1],
						tmp[lid][2],
						tmp[lid][3],
						tmp[lid][4],
						tmp[lid][5],
						tmp[lid][6],
						tmp[lid][7]
						);

				for (int t = 0; t < newCount; t++)
				{
					tmp[lid][t] = tmp[lid][t + 8];
				}
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int localIdx = min(15, counters[lid]);
	int cnt = min((int)atomic_add(indexes + lid, 8), (int)(DUCK_A_EDGES_64 - 8));
	int idx = (lid * DUCK_A_EDGES_64 + cnt) / 8;
	buffer[idx] = (uint8)
		(
			select((uint)0, tmp[lid][0], localIdx > 0),
			select((uint)0, tmp[lid][1], localIdx > 1),
			select((uint)0, tmp[lid][2], localIdx > 2),
			select((uint)0, tmp[lid][3], localIdx > 3),
			select((uint)0, tmp[lid][4], localIdx > 4),
			select((uint)0, tmp[lid][5], localIdx > 5),
			select((uint)0, tmp[lid][6], localIdx > 6),
			0
			);
}



__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel  void FluffySeed1B(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, const __global uint * source, __global uint8 * destination, __global const int * sourceIndexes, __global int * destinationIndexes, int startBlock)
{
	const int lid = get_local_id(0);
	const int group = get_group_id(0);

	__local uint tmp[64][15];
	__local int counters[64];

	counters[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	const int offsetMem = startBlock * DUCK_A_EDGES_64;
	const int myBucket = group / BKTGRAN;
	const int microBlockNo = group % BKTGRAN;
	const int bucketEdges = min(sourceIndexes[myBucket + startBlock], (int)(DUCK_A_EDGES_64));
	const int microBlockEdgesCount = (DUCK_A_EDGES_64 / BKTGRAN);
	const int loops = (microBlockEdgesCount / 64);

	for (int i = 0; i < loops; i++)
	{
		int edgeIndex = (microBlockNo * microBlockEdgesCount) + (64 * i) + lid;

		//if (edgeIndex < bucketEdges)
		{
			//uint nonce = source[offsetMem + (myBucket * DUCK_A_EDGES_64) + edgeIndex];
			uint nonce = (edgeIndex < bucketEdges) ? ld_cs_u32(source + (offsetMem + (myBucket * DUCK_A_EDGES_64) + edgeIndex)) : 0;
			
			bool skip = (nonce == 0) || (edgeIndex >= bucketEdges);

			uint edge = skip ? 0 : dipnode(v0i, v1i, v2i, v3i, nonce, 0);

			int bucket = (edge >> 6) & (64 - 1);

			barrier(CLK_LOCAL_MEM_FENCE);

			if (!skip)
			{
				int counter = min((int)atomic_inc(counters + bucket), (int)14);

				tmp[bucket][counter] = nonce;
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			int localIdx = min(15, counters[lid]);

			if (localIdx >= 8)
			{
				int newCount = (localIdx - 8);
				counters[lid] = newCount;
				int cnt = min((int)atomic_add(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 8), (int)(DUCK_A_EDGES - 8));

				int idx = ((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 8;

				destination[idx] = (uint8)
					(
						tmp[lid][0],
						tmp[lid][1],
						tmp[lid][2],
						tmp[lid][3],
						tmp[lid][4],
						tmp[lid][5],
						tmp[lid][6],
						tmp[lid][7]
						);

				for (int t = 0; t < newCount; t++)
				{
					tmp[lid][t] = tmp[lid][t + 8];
				}
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int localIdx = min(15, counters[lid]);
	int cnt = min((int)atomic_add(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 8), (int)(DUCK_A_EDGES - 8));
	int idx = ((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 8;
	destination[idx] = (uint8)
		(
			select((uint)0, tmp[lid][0], localIdx > 0),
			select((uint)0, tmp[lid][1], localIdx > 1),
			select((uint)0, tmp[lid][2], localIdx > 2),
			select((uint)0, tmp[lid][3], localIdx > 3),
			select((uint)0, tmp[lid][4], localIdx > 4),
			select((uint)0, tmp[lid][5], localIdx > 5),
			select((uint)0, tmp[lid][6], localIdx > 6),
			0
			);

}


#define FRB_T 1024
__attribute__((reqd_work_group_size(FRB_T, 1, 1)))
__kernel void FluffyRoundA(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, const __global uint * source, __global uint2 * destination, __global const int * sourceIndexes, __global int * destinationIndexes, const int bktInSize, const int bktOutSize)
{
	const int gid = get_global_id(0);
	const short lid = get_local_id(0);
	const int group = get_group_id(0);

	__local u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + FRB_T) / FRB_T;

	for (int i = 0; i < (8192 / FRB_T); i++)
		ecounters[lid + (FRB_T * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * FRB_T) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;

			//uint nonce = source[index];
			uint nonce = ld_nc_u32(source + index);

			if (nonce == 0) continue;

			uint hash = dipnode(v0i, v1i, v2i, v3i, nonce, 0);

			Increase2bCounter(ecounters, (hash & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = loops - 1; i >= 0; i--)
	{
		const int lindex = (i * FRB_T) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;

			uint nonce = ld_nc_u32(source + index);

			if (nonce == 0) continue;

			uint2 hash = dipnode2(v0i, v1i, v2i, v3i, nonce);

			if (Read2bCounter(ecounters, (hash.x & EDGEMASK) >> 12))
			{
				const int bucket = hash.y & BKTMASK4K;
				const int bktIdx = min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1);
				destination[(bucket * bktOutSize) + bktIdx] = (uint2)(hash.y, hash.x);
			}
		}
	}
}

__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void FluffyRoundB(const __global uint2 * source, __global uint2 * destination, __global const int * sourceIndexes, __global int * destinationIndexes, int bktInSize, int bktOutSize)
{
	const int gid = get_global_id(0);
	const short lid = get_local_id(0);
	const int group = get_group_id(0);

	__local u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + 1024) / 1024;

	for (int i = 0; i < 8; i++)
		ecounters[lid + (1024 * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * 1024) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;

			uint2 edge = ld_nc_u32_v2(source + index);

			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = loops - 1; i >= 0; i--)
	{
		const int lindex = (i * 1024) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;

			uint2 edge = ld_nc_u32_v2(source + index);


			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1);
				destination[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}

}



void Increase2bCounter_global(__local u32 * cache, __global u32 * ecounters, const int bucket)
{
	u32 cachePayload = bucket >> 11;
	u32 cacheIdx = bucket & 2047;
	u32 word = cacheIdx >> 2;
	u32 offset = ((cacheIdx & 3) << 3);
	u32 mask = 128u << offset;
	u32 mask2 = 64u << offset;

	u32 old = atomic_or(cache + word, mask);
	bool oldLock = (old & mask) > 0;
	bool oldMaxC = (old & mask2) > 0;
	u32 oldPayload = (old >> offset) & 63;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (!oldLock)
		atomic_or(cache + word, (u32)cachePayload << offset);

	barrier(CLK_LOCAL_MEM_FENCE);

	old = cache[word];
	oldMaxC = (old & mask2) > 0;
	oldPayload = (old >> offset) & 63;

	if (oldLock && (oldPayload == cachePayload))
	{
		atomic_or(cache + word, mask2);
	}
	else if (oldLock && (oldPayload != cachePayload))
	{
		int word = bucket >> 5;
		unsigned char bit = bucket & 0x1F;
		u32 mask = 1 << bit;

		u32 old = atomic_or(ecounters + word, mask) & mask;

		if (old > 0)
			atomic_or(ecounters + word + 4096, mask);
	}
}

bool Read2bCounter_global(__local u32 * cache, __global u32 * ecounters, const int bucket)
{
	u32 cachePayload = bucket >> 11;
	u32 cacheIdx = bucket & 2047;
	u32 word = cacheIdx / 4;
	u32 offset = ((cacheIdx & 3) << 3);
	u32 mask2 = 64 << offset;

	u32 old = cache[word];
	bool oldMaxC = (old & mask2) > 0;
	u32 oldPayload = (old >> offset) & 63;

	if (oldPayload == cachePayload)
	{
		return oldMaxC;
	}
	else
	{
		int word = bucket >> 5;
		unsigned char bit = bucket & 0x1F;
		u32 mask = 1 << bit;

		return (ecounters[word + 4096] & mask) > 0;
	}
}


__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void FluffyRoundC(const __global uint2 * source, __global uint2 * destination, __global const int * sourceIndexes, __global int * destinationIndexes, int bktInSize, int bktOutSize, __global uint * sp)
{
	const int gid = get_global_id(0);
	const short lid = get_local_id(0);
	const int group = get_group_id(0);

	__local u32 cache[512];
	__global u32 * ecounters = sp + group * 8192;

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + 64) / 64;

	for (int i = 0; i < 128; i++)
		ecounters[lid + (64 * i)] = 0;

	for (int i = 0; i < 32; i++)
		cache[lid + (64 * i)] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * 64) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;

			uint2 edge = ld_nc_u32_v2(source + index);

			Increase2bCounter_global(cache, ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = loops - 1; i >= 0; i--)
	{
		const int lindex = (i * 64) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;

			uint2 edge = ld_nc_u32_v2(source + index);

			if (Read2bCounter_global(cache, ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min(atomic_inc(destinationIndexes + bucket), bktOutSize - 1);
				destination[(bucket * bktOutSize) + bktIdx] = (uint2)(edge.y, edge.x);
			}
		}
	}

}
