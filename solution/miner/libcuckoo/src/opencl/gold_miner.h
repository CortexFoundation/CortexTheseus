#ifndef GOLD_MINER_H
#define GOLD_MINER_H
#include <string>
static const std::string kernel_source = R"(
// Cuckaroo Cycle, a memory-hard proof-of-work by John Tromp and team Grin
// Copyright (c) 2018 Jiri Photon Vadura and John Tromp
// This GGM miner file is covered by the FAIR MINING license

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef uint8 u8;
typedef uint16 u16;
typedef uint u32;
typedef ulong u64;

typedef u32 node_t;
typedef u64 nonce_t;

//#define EDGEBITS 29
//#define PROOFSIZE 42

// number of edges
#define NEDGES ((node_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK (NEDGES - 1)

#define CTHREADS 1024
#define BKTMASK4K (4096-1)
#define BKTGRAN 32

#define DUCK_SIZE_A 129L
#define DUCK_SIZE_B 83L

#define XBITS 6
#define YBITS 6
#define ZBITS (EDGEBITS - 12)
#define SUB_BUCKET_SIZE (1 << (ZBITS - 7))

#define DUCK_A_EDGES (DUCK_SIZE_A * SUB_BUCKET_SIZE)
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64L)

#define DUCK_B_EDGES (DUCK_SIZE_B * SUB_BUCKET_SIZE)
#define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64L)

#define EDGE_BLOCK_SIZE (64)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)


#define SIPROUND \
	  do { \
	      v0 += v1; v2 += v3; v1 = rotate(v1,(ulong)13); \
	      v3 = rotate(v3,(ulong)16); v1 ^= v0; v3 ^= v2; \
	      v0 = rotate(v0,(ulong)32); v2 += v1; v0 += v3; \
	      v1 = rotate(v1,(ulong)17);   v3 = rotate(v3,(ulong)21); \
	      v1 ^= v2; v3 ^= v0; v2 = rotate(v2,(ulong)32); \
	    } while(0)

void Increase2bCounter (__local u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	u32 old = atomic_or (ecounters + word, mask) & mask;

	if (old > 0)
		atomic_or (ecounters + word + 4096, mask);
}

bool Read2bCounter (__local u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	return (ecounters[word + 4096] & mask) > 0;
}

inline u32 dipnode(const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, const u32 uorv, const u64 nce){
	u64 nonce = 2 * nce + uorv;
	u64 v0 = v0i, v1 = v1i, v2 = v2i, v3 =v3i ^ nonce;	
	for(int i = 0; i < 2; i++)
		SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	for(int i = 0; i < 4; i++)
		SIPROUND;
	return (v0 ^ v1 ^ v2 ^ v3) & EDGEMASK;
}

__kernel void Cuckoo_FluffySeed2A (const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, __global ulong4 * bufferA, __global ulong4 * bufferB, __global u32 * indexes)
{
	const int gid = get_global_id (0);
	const short lid = get_local_id (0);

	__global ulong4 *buffer;
	__local u64 tmp[64][16];
	__local u32 counters[64];

	if (lid < 64)
		counters[lid] = 0;

	barrier (CLK_LOCAL_MEM_FENCE);
	const int loops = NEDGES / (get_global_size(0));

	for (int i = 0; i < loops; i++)
	{
		u64 nonce = gid * loops + i;
		u32 node0 = dipnode(v0i, v1i, v2i, v3i, 0, nonce);
		u32 node1 = dipnode(v0i, v1i, v2i, v3i, 1, nonce);
		int bucket = node0 & 63;

		int counter = atomic_add (counters + bucket, (u32) 1);
		int counterLocal = counter % 16;
		tmp[bucket][counterLocal] = node0 |  ((u64)node1 <<32);

		barrier (CLK_LOCAL_MEM_FENCE);

		if ((counter > 0) && (counterLocal == 0 || counterLocal == 8))
		{
			int cnt = min ((int) atomic_add (indexes + bucket, 8), (int) (DUCK_A_EDGES_64 - 8));
			int idx = ((bucket < 32 ? bucket : bucket - 32) * DUCK_A_EDGES_64 + cnt) / 4;
			buffer = bucket < 32 ? bufferA : bufferB;

			buffer[idx] = (ulong4) (atom_xchg (&tmp[bucket][8 - counterLocal], (u64) 0),
				atom_xchg (&tmp[bucket][9 - counterLocal], (u64) 0), atom_xchg (&tmp[bucket][10 - counterLocal], (u64) 0), atom_xchg (&tmp[bucket][11 - counterLocal], (u64) 0));
			buffer[idx + 1] = (ulong4) (atom_xchg (&tmp[bucket][12 - counterLocal], (u64) 0),
				atom_xchg (&tmp[bucket][13 - counterLocal], (u64) 0), atom_xchg (&tmp[bucket][14 - counterLocal], (u64) 0), atom_xchg (&tmp[bucket][15 - counterLocal], (u64) 0));
		}

	}

	barrier (CLK_LOCAL_MEM_FENCE);

	if (lid < 64)
	{
		int counter = counters[lid];
		int counterBase = (counter % 16) >= 8 ? 8 : 0;
		int counterCount = (counter % 8);
		for (int i = 0; i < (8 - counterCount); i++)
			tmp[lid][counterBase + counterCount + i] = 0;
		int cnt = min ((int) atomic_add (indexes + lid, 8), (int) (DUCK_A_EDGES_64 - 8));
		int idx = ((lid < 32 ? lid : lid - 32) * DUCK_A_EDGES_64 + cnt) / 4;
		buffer = lid < 32 ? bufferA : bufferB;
		buffer[idx] = (ulong4) (tmp[lid][counterBase], tmp[lid][counterBase + 1], tmp[lid][counterBase + 2], tmp[lid][counterBase + 3]);
		buffer[idx + 1] = (ulong4) (tmp[lid][counterBase + 4], tmp[lid][counterBase + 5], tmp[lid][counterBase + 6], tmp[lid][counterBase + 7]);
	}

}
//__attribute__ ((reqd_work_group_size (128, 1, 1)))
__kernel void FluffySeed2A (const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, __global ulong4 * bufferA, __global ulong4 * bufferB, __global u32 * indexes)
{
	const int gid = get_global_id (0);
	const short lid = get_local_id (0);

	__global ulong4 *buffer;
	__local u64 tmp[64][16];
	__local u32 counters[64];
//	u64 sipblock[64];

	u64 v0;
	u64 v1;
	u64 v2;
	u64 v3;

	if (lid < 64)
		counters[lid] = 0;

	barrier (CLK_LOCAL_MEM_FENCE);
	const int loops = NEDGES / (get_global_size(0));

	for (int i = 0; i < loops; i += EDGE_BLOCK_SIZE)
	{
		u64 blockNonce = gid * loops + i;

		v0 = v0i;
		v1 = v1i;
		v2 = v2i;
		v3 = v3i;

		for (u32 b = 0; b < EDGE_BLOCK_SIZE; b++)
		{
			v3 ^= blockNonce + b;
			for (int r = 0; r < 2; r++)
				SIPROUND;
			v0 ^= blockNonce + b;
			v2 ^= 0xff;
			for (int r = 0; r < 4; r++)
				SIPROUND;

//			sipblock[b] = (v0 ^ v1) ^ (v2 ^ v3);

		}
//		u64 last = sipblock[EDGE_BLOCK_MASK];
		u64 last = (v0 ^ v1) ^ (v2 ^ v3);
		v0 = v0i; v1 = v1i; v2 = v2i; v3 = v3i;

		for (short s = 0; s < EDGE_BLOCK_SIZE; s++)
		{
			u64 lookup;
			if(s == EDGE_BLOCK_MASK) lookup = last;
			else{
				v3 ^= blockNonce + s;
				for (int r = 0; r < 2; r++)
					SIPROUND;
				v0 ^= blockNonce + s;
				v2 ^= 0xff;
				for (int r = 0; r < 4; r++)
					SIPROUND;
				lookup = ((v0 ^ v1) ^ (v2 ^ v3)) ^ last;
			}
			//u64 lookup = s == EDGE_BLOCK_MASK ? last : sipblock[s] ^ last;
			uint2 hash = (uint2) (lookup & EDGEMASK, (lookup >> 32) & EDGEMASK);
			int bucket = hash.x & 63;

			//barrier (CLK_LOCAL_MEM_FENCE);

			int counter = atomic_add (counters + bucket, (u32) 1);
			int counterLocal = counter % 16;
			tmp[bucket][counterLocal] = hash.x | ((u64) hash.y << 32);

			barrier (CLK_LOCAL_MEM_FENCE);

			if ((counter > 0) && (counterLocal == 0 || counterLocal == 8))
			{
				int cnt = min ((int) atomic_add (indexes + bucket, 8), (int) (DUCK_A_EDGES_64 - 8));
				int idx = ((bucket < 32 ? bucket : bucket - 32) * DUCK_A_EDGES_64 + cnt) / 4;
				buffer = bucket < 32 ? bufferA : bufferB;

				buffer[idx] = (ulong4) (atom_xchg (&tmp[bucket][8 - counterLocal], (u64) 0),
					atom_xchg (&tmp[bucket][9 - counterLocal], (u64) 0), atom_xchg (&tmp[bucket][10 - counterLocal], (u64) 0), atom_xchg (&tmp[bucket][11 - counterLocal], (u64) 0));
				buffer[idx + 1] = (ulong4) (atom_xchg (&tmp[bucket][12 - counterLocal], (u64) 0),
					atom_xchg (&tmp[bucket][13 - counterLocal], (u64) 0), atom_xchg (&tmp[bucket][14 - counterLocal], (u64) 0), atom_xchg (&tmp[bucket][15 - counterLocal], (u64) 0));
			}

		}
	}

	barrier (CLK_LOCAL_MEM_FENCE);

	if (lid < 64)
	{
		int counter = counters[lid];
		int counterBase = (counter % 16) >= 8 ? 8 : 0;
		int counterCount = (counter % 8);
		for (int i = 0; i < (8 - counterCount); i++)
			tmp[lid][counterBase + counterCount + i] = 0;
		int cnt = min ((int) atomic_add (indexes + lid, 8), (int) (DUCK_A_EDGES_64 - 8));
		int idx = ((lid < 32 ? lid : lid - 32) * DUCK_A_EDGES_64 + cnt) / 4;
		buffer = lid < 32 ? bufferA : bufferB;
		buffer[idx] = (ulong4) (tmp[lid][counterBase], tmp[lid][counterBase + 1], tmp[lid][counterBase + 2], tmp[lid][counterBase + 3]);
		buffer[idx + 1] = (ulong4) (tmp[lid][counterBase + 4], tmp[lid][counterBase + 5], tmp[lid][counterBase + 6], tmp[lid][counterBase + 7]);
	}

}

//__attribute__ ((reqd_work_group_size (128, 1, 1)))
__kernel void FluffySeed2B (const __global uint2 * source, __global ulong4 * destination1, __global ulong4 * destination2, const __global int *sourceIndexes, __global int *destinationIndexes,
	int startBlock)
{
	const int lid = get_local_id (0);
	const int group = get_group_id (0);
	const int group_size = get_local_size(0);

	__global ulong4 *destination = destination1;
	__local u64 tmp[64][16];
	__local int counters[64];

	if (lid < 64)
		counters[lid] = 0;

	barrier (CLK_LOCAL_MEM_FENCE);

	int offsetMem = startBlock * DUCK_A_EDGES_64;
	int offsetBucket = 0;
	const int myBucket = group / BKTGRAN;
	const int microBlockNo = group % BKTGRAN;
	const int bucketEdges = min (sourceIndexes[myBucket + startBlock], (int) (DUCK_A_EDGES_64));
	const int microBlockEdgesCount = (DUCK_A_EDGES_64 / BKTGRAN);
	const int loops = (microBlockEdgesCount / group_size);

	if ((startBlock == 32) && (myBucket >= 30))
	{
		offsetMem = 0;
		destination = destination2;
		offsetBucket = 30;
	}

	for (int i = 0; i < loops; i++)
	{
		int edgeIndex = (microBlockNo * microBlockEdgesCount) + (group_size * i) + lid;

		{
			uint2 edge = source[ /*offsetMem + */ (myBucket * DUCK_A_EDGES_64) + edgeIndex];
			bool skip = (edgeIndex >= bucketEdges) || (edge.x == 0 && edge.y == 0);

			int bucket = (edge.x >> 6) & (64 - 1);

			barrier (CLK_LOCAL_MEM_FENCE);

			int counter = 0;
			int counterLocal = 0;

			if (!skip)
			{
				counter = atomic_add (counters + bucket, (u32) 1);
				counterLocal = counter % 16;
				tmp[bucket][counterLocal] = edge.x | ((u64) edge.y << 32);
			}

			barrier (CLK_LOCAL_MEM_FENCE);

			if ((counter > 0) && (counterLocal == 0 || counterLocal == 8))
			{
				int cnt = min ((int) atomic_add (destinationIndexes + startBlock * 64 + myBucket * 64 + bucket, 8), (int) (DUCK_A_EDGES - 8));
				int idx = (offsetMem + (((myBucket - offsetBucket) * 64 + bucket) * DUCK_A_EDGES + cnt)) / 4;

				destination[idx] = (ulong4) (atom_xchg (&tmp[bucket][8 - counterLocal], 0),
					atom_xchg (&tmp[bucket][9 - counterLocal], 0), atom_xchg (&tmp[bucket][10 - counterLocal], 0), atom_xchg (&tmp[bucket][11 - counterLocal], 0));
				destination[idx + 1] = (ulong4) (atom_xchg (&tmp[bucket][12 - counterLocal], 0),
					atom_xchg (&tmp[bucket][13 - counterLocal], 0), atom_xchg (&tmp[bucket][14 - counterLocal], 0), atom_xchg (&tmp[bucket][15 - counterLocal], 0));
			}
		}
	}

	barrier (CLK_LOCAL_MEM_FENCE);

	if (lid < 64)
	{
		int counter = counters[lid];
		int counterBase = (counter % 16) >= 8 ? 8 : 0;
		int cnt = min ((int) atomic_add (destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 8), (int) (DUCK_A_EDGES - 8));
		int idx = (offsetMem + (((myBucket - offsetBucket) * 64 + lid) * DUCK_A_EDGES + cnt)) / 4;
		destination[idx] = (ulong4) (tmp[lid][counterBase], tmp[lid][counterBase + 1], tmp[lid][counterBase + 2], tmp[lid][counterBase + 3]);
		destination[idx + 1] = (ulong4) (tmp[lid][counterBase + 4], tmp[lid][counterBase + 5], tmp[lid][counterBase + 6], tmp[lid][counterBase + 7]);
	}
}

//__attribute__ ((reqd_work_group_size (256, 1, 1)))
__kernel void FluffyRound1 (const __global uint2 * source1, const __global uint2 * source2, __global uint2 * destination, const __global int *sourceIndexes, __global int *destinationIndexes,
	const int bktInSize, const int bktOutSize)
{
	const int lid = get_local_id (0);
	const int group = get_group_id (0);
	const int group_size = get_local_size(0);

	const __global uint2 *source = group < (62 * 64) ? source1 : source2;
	int groupRead = group < (62 * 64) ? group : group - (62 * 64);

	__local u32 ecounters[8192];

	const int edgesInBucket = min (sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + group_size) / group_size;

	for (int i = 0; i < 8192/group_size; i++)
		ecounters[lid + (group_size * i)] = 0;

	barrier (CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * group_size) + lid;

		if (lindex < edgesInBucket)
		{

			const int index = (bktInSize * groupRead) + lindex;

			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0)
				continue;

			Increase2bCounter (ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier (CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * group_size) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * groupRead) + lindex;

			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0)
				continue;

			if (Read2bCounter (ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min (atomic_add (destinationIndexes + bucket, 1), bktOutSize - 1);
				destination[(bucket * bktOutSize) + bktIdx] = (uint2) (edge.y, edge.x);
			}
		}
	}

}

//__attribute__ ((reqd_work_group_size (256, 1, 1)))
__kernel void FluffyTail (const __global uint2 * source, __global uint2 * destination, const __global int *sourceIndexes, __global int *destinationIndexes)
{
	const int lid = get_local_id (0);
	const int group = get_group_id (0);

	int myEdges = sourceIndexes[group];
	__local int destIdx;

	if (lid == 0)
		destIdx = atomic_add (destinationIndexes, myEdges);

	barrier (CLK_LOCAL_MEM_FENCE);

	if (lid < myEdges)
	{
		destination[destIdx + lid] = source[group * DUCK_B_EDGES + lid];
	}
}

__kernel void Cuckoo_FluffyRecovery (const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, const __constant u64 * recovery, __global int *indexes)
{
	const int gid = get_global_id (0);
	const short lid = get_local_id (0);

	__local u32 nonces[PROOFSIZE];
	u64 sipblock[64];


	if (lid < PROOFSIZE)
		nonces[lid] = 0;

	barrier (CLK_LOCAL_MEM_FENCE);

	const int loops = NEDGES / get_global_size(0);
	for (int i = 0; i < loops; i += 1)
	{
		u64 nonce = gid * loops + i;
		u32 node0 = dipnode(v0i, v1i, v2i, v3i, 0, nonce);
		u32 node1 = dipnode(v0i, v1i, v2i, v3i, 1, nonce);

		u64 u = node0;//lookup & EDGEMASK;
		u64 v = node1;//(lookup >> 32) & EDGEMASK;

		u64 a = u | (v << 32);
		u64 b = v | (u << 32);

		for (int i = 0; i < PROOFSIZE; i++)
		{
			if ((recovery[i] == a) || (recovery[i] == b))
				nonces[i] = nonce;
		}
	}

	barrier (CLK_LOCAL_MEM_FENCE);

	if (lid < PROOFSIZE)
	{
		if (nonces[lid] > 0)
			indexes[lid] = nonces[lid];
	}
}

//__attribute__ ((reqd_work_group_size (256, 1, 1)))
__kernel void FluffyRecovery (const u64 v0i, const u64 v1i, const u64 v2i, const u64 v3i, const __constant u64 * recovery, __global int *indexes)
{
	const int gid = get_global_id (0);
	const short lid = get_local_id (0);

	__local u32 nonces[PROOFSIZE];
	u64 sipblock[64];

	u64 v0;
	u64 v1;
	u64 v2;
	u64 v3;

	if (lid < PROOFSIZE)
		nonces[lid] = 0;

	barrier (CLK_LOCAL_MEM_FENCE);

	const int loops = NEDGES / get_global_size(0);
	for (int i = 0; i < loops; i += EDGE_BLOCK_SIZE)
	{
		u64 blockNonce = gid * loops + i;

		v0 = v0i;
		v1 = v1i;
		v2 = v2i;
		v3 = v3i;

		for (u32 b = 0; b < EDGE_BLOCK_SIZE; b++)
		{
			v3 ^= blockNonce + b;
			SIPROUND;
			SIPROUND;
			v0 ^= blockNonce + b;
			v2 ^= 0xff;
			SIPROUND;
			SIPROUND;
			SIPROUND;
			SIPROUND;

			sipblock[b] = (v0 ^ v1) ^ (v2 ^ v3);

		}
		const u64 last = sipblock[EDGE_BLOCK_MASK];

		for (short s = EDGE_BLOCK_MASK; s >= 0; s--)
		{
			u64 lookup = s == EDGE_BLOCK_MASK ? last : sipblock[s] ^ last;
			u64 u = lookup & EDGEMASK;
			u64 v = (lookup >> 32) & EDGEMASK;

			u64 a = u | (v << 32);
			u64 b = v | (u << 32);

			for (int i = 0; i < PROOFSIZE; i++)
			{
				if ((recovery[i] == a) || (recovery[i] == b))
					nonces[i] = blockNonce + s;
			}
		}
	}

	barrier (CLK_LOCAL_MEM_FENCE);

	if (lid < PROOFSIZE)
	{
		if (nonces[lid] > 0)
			indexes[lid] = nonces[lid];
	}
}

#define BKT_OFFSET 255
#define BKT_STEP 32
//__attribute__ ((reqd_work_group_size (256, 1, 1)))
__kernel void FluffyRoundNO1 (const __global uint2 * source, __global uint2 * destination, const __global int *sourceIndexes, __global int *destinationIndexes)
{
	const int lid = get_local_id (0);
	const int group = get_group_id (0);
	const int group_size = get_local_size(0);

	const int bktInSize = DUCK_B_EDGES;
	const int bktOutSize = DUCK_B_EDGES;

	__local u32 ecounters[8192];

	const int edgesInBucket = min (sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + group_size) / group_size;

	for (int i = 0; i < 8192/group_size; i++)
		ecounters[lid + (group_size * i)] = 0;

	barrier (CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * group_size) + lid;

		if (lindex < edgesInBucket)
		{

			const int index = (bktInSize * group) + lindex;

			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0)
				continue;

			Increase2bCounter (ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier (CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * group_size) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = (bktInSize * group) + lindex;

			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0)
				continue;

			if (Read2bCounter (ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min (atomic_add (destinationIndexes + bucket, 1), bktOutSize - 1 - ((bucket & BKT_OFFSET) * BKT_STEP));
				destination[((bucket & BKT_OFFSET) * BKT_STEP) + (bucket * bktOutSize) + bktIdx] = (uint2) (edge.y, edge.x);
			}
		}
	}

}

//__attribute__ ((reqd_work_group_size (256, 1, 1)))
__kernel void FluffyRoundNON (const __global uint2 * source, __global uint2 * destination, const __global int *sourceIndexes, __global int *destinationIndexes)
{
	const int lid = get_local_id (0);
	const int group = get_group_id (0);
	const int group_size = get_local_size(0);

	const int bktInSize = DUCK_B_EDGES;
	const int bktOutSize = DUCK_B_EDGES;

	__local u32 ecounters[8192];

	const int edgesInBucket = min (sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + group_size) / group_size;

	for (int i = 0; i < 8192/group_size; i++)
		ecounters[lid + (group_size * i)] = 0;

	barrier (CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * group_size) + lid;

		if (lindex < edgesInBucket)
		{

			const int index = ((group & BKT_OFFSET) * BKT_STEP) + (bktInSize * group) + lindex;

			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0)
				continue;

			Increase2bCounter (ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	barrier (CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < loops; i++)
	{
		const int lindex = (i * group_size) + lid;

		if (lindex < edgesInBucket)
		{
			const int index = ((group & BKT_OFFSET) * BKT_STEP) + (bktInSize * group) + lindex;

			uint2 edge = source[index];

			if (edge.x == 0 && edge.y == 0)
				continue;

			if (Read2bCounter (ecounters, (edge.x & EDGEMASK) >> 12))
			{
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min (atomic_add (destinationIndexes + bucket, 1), bktOutSize - 1 - ((bucket & BKT_OFFSET) * BKT_STEP));
				destination[((bucket & BKT_OFFSET) * BKT_STEP) + (bucket * bktOutSize) + bktIdx] = (uint2) (edge.y, edge.x);
			}
		}
	}

}

//__attribute__ ((reqd_work_group_size (256, 1, 1)))
__kernel void FluffyTailO (const __global uint2 * source, __global uint2 * destination, const __global int *sourceIndexes, __global int *destinationIndexes)
{
	const int lid = get_local_id (0);
	const int group = get_group_id (0);

	int myEdges = sourceIndexes[group];
	__local int destIdx;

	if (lid == 0)
		destIdx = atomic_add (destinationIndexes, myEdges);

	barrier (CLK_LOCAL_MEM_FENCE);

	if (lid < myEdges)
	{
		destination[destIdx + lid] = source[((group & BKT_OFFSET) * BKT_STEP) + group * DUCK_B_EDGES + lid];
	}
}
)";

inline std::string get_kernel_source(){
	return kernel_source;
}
#endif

