#include <time.h>

#include "trimmer.h"
#include "../siphash.cuh"
namespace cuckoogpu
{

#define TROMP_SEEDA
#define TROMP_SEEDB
#define TROMP_ROUND
#define TROMP_TAIL

    //#define TIMER

#define DUCK_A_EDGES (EDGES_A)
#define DUCK_A_EDGES_NX (DUCK_A_EDGES * NX)
#define DUCK_B_EDGES (EDGES_B)
#define DUCK_B_EDGES_NX (DUCK_B_EDGES * NX)


    __device__ u32 endpoint (const siphash_keys & sipkeys, u32 nonce, int uorv)
    {
        return dipnode (sipkeys, nonce, uorv);
    }


    __device__ uint2 make_Edge (const u32 nonce, const uint2 dummy, const u32 node0, const u32 node1)
    {
        return make_uint2 (node0, node1);
    }

    __device__ uint2 make_Edge (const uint2 edge, const uint2 dummy, const u32 node0, const u32 node1)
    {
        return edge;
    }

    __device__ u32 make_Edge (const u32 nonce, const u32 dummy, const u32 node0, const u32 node1)
    {
        return nonce;
    }
    __device__ ulonglong4 Pack4edges (const uint2 e1, const uint2 e2, const uint2 e3, const uint2 e4)
    {
        u64 r1 = (((u64) e1.y << 32) | ((u64) e1.x));
        u64 r2 = (((u64) e2.y << 32) | ((u64) e2.x));
        u64 r3 = (((u64) e3.y << 32) | ((u64) e3.x));
        u64 r4 = (((u64) e4.y << 32) | ((u64) e4.x));
        return make_ulonglong4 (r1, r2, r3, r4);
    }
    // ===== Above =======

    __device__ __forceinline__ void Increase2bCounter (u32 * ecounters, const int bucket)
    {
        int word = bucket >> 5;
        unsigned char bit = bucket & 0x1F;
        u32 mask = 1 << bit;

        u32 old = atomicOr (ecounters + word, mask) & mask;
        if (old)
            atomicOr (ecounters + word + NZ / 32, mask);
    }

    __device__ __forceinline__ bool Read2bCounter (u32 * ecounters, const int bucket)
    {
        int word = bucket >> 5;
        unsigned char bit = bucket & 0x1F;
        u32 mask = 1 << bit;

        return (ecounters[word + NZ / 32] & mask) != 0;
    }
    __device__ __forceinline__  void bitmapset(u32 *ebitmap, const int bucket) {
        int word = bucket >> 5;
        unsigned char bit = bucket & 0x1F;
        u32 mask = 1 << bit;
        u32 old = atomicOr(ebitmap + word, mask) & mask;
        if(old) atomicOr(ebitmap + word + NZ/32, mask);
    }

    __device__ __forceinline__  bool bitmaptest(u32 *ebitmap, const int bucket) {
        int word = bucket >> 5;
        unsigned char bit = bucket & 0x1F;
        return (ebitmap[word + NZ/32] >> bit) & 1;
    }

    __device__ bool null (u32 nonce)
    {
        return nonce == 0;
    }

    __device__ bool null (uint2 nodes)
    {
        return nodes.x == 0 && nodes.y == 0;
    }

    __device__ u64 dipblock (const siphash_keys & keys, const edge_t edge, u64 * buf)
    {
        diphash_state shs (keys);
        edge_t edge0 = edge & ~EDGE_BLOCK_MASK;
        u32 i;
        for (i = 0; i < EDGE_BLOCK_MASK; i++)
        {
            shs.hash24 (edge0 + i);
            buf[i] = shs.xor_lanes ();
        }
        shs.hash24 (edge0 + i);
        buf[i] = 0;
        return shs.xor_lanes ();
    }
    __device__ u32 endpoint (uint2 nodes, int uorv)
    {
        return uorv ? nodes.y : nodes.x;
    }
    __device__ u32 endpoint (const siphash_keys & sipkeys, uint2 nodes, int uorv)
    {
        return uorv ? nodes.y : nodes.x;
    }

    __constant__ uint2 recoveredges[PROOFSIZE];

    __global__ void Cuckaroo_Recovery (const siphash_keys & sipkeys, int *indexes)
    {
        const int gid = blockDim.x * blockIdx.x + threadIdx.x;
        const int lid = threadIdx.x;
        const int nthreads = blockDim.x * gridDim.x;
        const int loops = NEDGES / nthreads;
        __shared__ u32 nonces[PROOFSIZE];

        if (lid < PROOFSIZE)
            nonces[lid] = 0;
        __syncthreads ();
        for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE)
        {
            u32 nonce0 = gid * loops + blk;
            diphash_state shs(sipkeys);
            edge_t edge0 = nonce0 & ~EDGE_BLOCK_MASK;
            u32 i;
            for(i = 0; i < EDGE_BLOCK_SIZE; i++){
                shs.hash24(edge0 + i);
            }
            const u64 last = shs.xor_lanes();

            diphash_state shs2(sipkeys);
            for (i = 0; i < EDGE_BLOCK_SIZE; i++)
            {
                u64 edge;
                if(i == EDGE_BLOCK_MASK) edge = last;
                else{
                    shs2.hash24(edge0 + i);
                    edge = shs2.xor_lanes() ^ last;
                }
                u32 u = edge & EDGEMASK;
                u32 v = (edge >> 32) & EDGEMASK;
                for (int p = 0; p < PROOFSIZE; p++)
                {
                    if (recoveredges[p].x == u && recoveredges[p].y == v)
                        nonces[p] = nonce0 + i;
                }
            }
        }
        __syncthreads ();
        if (lid < PROOFSIZE)
        {
            if (nonces[lid] > 0){
                indexes[lid] = nonces[lid];
            }
        }
    }
	__global__ void Cuckoo_Recovery (const siphash_keys & sipkeys, int *indexes)
	{
		const int gid = blockDim.x * blockIdx.x + threadIdx.x;
		const int lid = threadIdx.x;
		const int nthreads = blockDim.x * gridDim.x;
		const int loops = NEDGES / nthreads;
		__shared__ u32 nonces[PROOFSIZE];

		if (lid < PROOFSIZE)
			nonces[lid] = 0;
		__syncthreads ();
		for (int i = 0; i < loops; i++)
		{
			u64 nonce = gid * loops + i;
			u64 u = dipnode (sipkeys, nonce, 0);
			u64 v = dipnode (sipkeys, nonce, 1);
			for (int i = 0; i < PROOFSIZE; i++)
			{
				if (recoveredges[i].x == u && recoveredges[i].y == v)
					nonces[i] = nonce;
			}
		}
		__syncthreads ();
		if (lid < PROOFSIZE)
		{
			if (nonces[lid] > 0)
				indexes[lid] = nonces[lid];
		}
	}



    template < int maxOut > __global__ void Cuckaroo_SeedA (const siphash_keys & sipkeys, ulonglong4* __restrict__ buffer, int *__restrict__ indexes)
    {
        const int group = blockIdx.x;
        const int dim = blockDim.x;
        const int lid = threadIdx.x;
        const int gid = group * dim + lid;
        const int nthreads = gridDim.x * dim;
        const int FLUSHA2 = 2 * FLUSHA;

        __shared__ uint2 tmp[NX][FLUSHA2];	// needs to be ulonglong4 aligned
        const int TMPPERLL4 = sizeof (ulonglong4) / sizeof (uint2);
        __shared__ int counters[NX];

#pragma unroll
        for (int row = lid; row < NX; row += dim)
            counters[row] = 0;
        __syncthreads ();

        const int col = group & YMASK;
        const int loops = NEDGES / nthreads;	// assuming THREADS_HAVE_EDGES checked
        for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE)
        {
            u32 nonce0 = gid * loops + blk;
            diphash_state shs(sipkeys);
            edge_t edge0 = nonce0 & ~EDGE_BLOCK_MASK;
            u32 i;
            for(i = 0; i < EDGE_BLOCK_SIZE; i++){
                shs.hash24(edge0 + i);
            }
            const u64 last = shs.xor_lanes();

            shs.set(sipkeys);
            u32 e;
            for (e = 0; e < EDGE_BLOCK_SIZE; e++)
            {
                u64 edge;
                if(e == EDGE_BLOCK_MASK) edge = last;
                else {
                    shs.hash24(edge0 + e);
                    edge = shs.xor_lanes() ^ last;
                }
                u32 node0 = edge & EDGEMASK;
                u32 node1 = (edge >> 32) & EDGEMASK;
                int row = node0 >> YZBITS;
                int counter = min ((int) atomicAdd (counters + row, 1), (int) (FLUSHA2 - 1));	// assuming ROWS_LIMIT_LOSSES checked
                tmp[row][counter] = make_uint2 (node0, node1);
                __syncthreads ();

                if (counter == FLUSHA - 1)
                {
                    int localIdx = min (FLUSHA2, counters[row]);
                    int newCount = localIdx % FLUSHA;
                    int nflush = localIdx - newCount;
                    u32 grp = row * NY + col;
                    int cnt = min ((int) atomicAdd (indexes + grp, nflush), (int) (maxOut - nflush));
#pragma unroll
                    for(int i = 0; i < FLUSHA; i += TMPPERLL4){
                        buffer[((u64) grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *) (&tmp[row][i]);
                    }
                    for (int i = FLUSHA; i < nflush; i += TMPPERLL4)
                    {
                        buffer[((u64) grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *) (&tmp[row][i]);
                    }
                    for (int t = 0; t < newCount; t++)
                    {
                        tmp[row][t] = tmp[row][t + nflush];
                    }
                    counters[row] = newCount;
                }
                __syncthreads ();
            }
        }
        uint2 zero = make_uint2 (0, 0);
        for (int row = lid; row < NX; row += dim)
        {
            int localIdx = min (FLUSHA2, counters[row]);
            u32 grp = row * NY + col;
            for (int j = localIdx; j % TMPPERLL4; j++)
                tmp[row][j] = zero;

            if (localIdx > 0)
            {
                int tmpl = (localIdx + TMPPERLL4 - 1) / TMPPERLL4 * TMPPERLL4;
                int cnt = min ((int) atomicAdd (indexes + grp, tmpl), (int) (maxOut - tmpl));
                for (int i = 0; i < localIdx; i += TMPPERLL4)
                {
                    //      int cnt = min((int)atomicAdd(indexes + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
                    buffer[((u64) grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *) (&tmp[row][i]);
                }
            }
        }
    }
    template < int maxOut, typename EdgeOut > __global__ void Cuckoo_SeedA (const siphash_keys & sipkeys, ulonglong4 * __restrict__ buffer, int *__restrict__ indexes)
    {
        const int group = blockIdx.x;
        const int dim = blockDim.x;
        const int lid = threadIdx.x;
        const int gid = group * dim + lid;
        const int nthreads = gridDim.x * dim;
        const int FLUSHA2 = 2 * FLUSHA;

        __shared__ EdgeOut tmp[NX][FLUSHA2];	// needs to be ulonglong4 aligned
        const int TMPPERLL4 = sizeof (ulonglong4) / sizeof (EdgeOut);
        __shared__ int counters[NX];

        for (int row = lid; row < NX; row += dim)
            counters[row] = 0;
        __syncthreads ();

        const int col = group % NX;
        const int loops = NEDGES / nthreads;
        for (int i = 0; i < loops; i++)
        {
            u32 nonce = gid * loops + i;
            u32 node1, node0 = dipnode (sipkeys, (u64) nonce, 0);
            if (sizeof (EdgeOut) == sizeof (uint2))
                node1 = dipnode (sipkeys, (u64) nonce, 1);
            int row = node0 >> YZBITS;
            int counter = min ((int) atomicAdd (counters + row, 1), (int) (FLUSHA2 - 1));
            tmp[row][counter] = make_Edge (nonce, tmp[0][0], node0, node1);
            __syncthreads ();
            if (counter == FLUSHA - 1)
            {
                int localIdx = min (FLUSHA2, counters[row]);
                int newCount = localIdx % FLUSHA;
                int nflush = localIdx - newCount;
                u32 grp = row * NX + col;
                int cnt = min ((int) atomicAdd (indexes + grp, nflush), (int) (maxOut - nflush));
                for (int i = 0; i < nflush; i += TMPPERLL4)
                    buffer[((u64) grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *) (&tmp[row][i]);
                for (int t = 0; t < newCount; t++)
                {
                    tmp[row][t] = tmp[row][t + nflush];
                }
                counters[row] = newCount;
            }
            __syncthreads ();
        }
        EdgeOut zero = make_Edge (0, tmp[0][0], 0, 0);
        for (int row = lid; row < NX; row += dim)
        {
            int localIdx = min (FLUSHA2, counters[row]);
            u32 grp = row * NX + col;
            for (int j = localIdx; j % TMPPERLL4; j++)
                tmp[row][j] = zero;

            if (localIdx > 0)
            {
                int tmpl = (localIdx + TMPPERLL4 - 1) / TMPPERLL4 * TMPPERLL4;
                int cnt = min ((int) atomicAdd (indexes + grp, tmpl), (int) (maxOut - tmpl));
                for (int i = 0; i < localIdx; i += TMPPERLL4)
                {
                    //            int cnt = min((int)atomicAdd(indexes + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
                    buffer[((u64) grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *) (&tmp[row][i]);
                }
            }
        }

    }

    template < int maxOut, typename EdgeOut >
        __global__ void SeedB (const siphash_keys & sipkeys, const EdgeOut * __restrict__ source, ulonglong4 * __restrict__ destination, const u32 * __restrict__ sourceIndexes,
                u32 * __restrict__ destinationIndexes)
        {
            const int group = blockIdx.x;
            const int dim = blockDim.x;
            const int lid = threadIdx.x;
            const int FLUSHB2 = 2 * FLUSHB;

            __shared__ EdgeOut tmp[NY][FLUSHB2];
            const int TMPPERLL4 = sizeof (ulonglong4) / sizeof (EdgeOut);
            __shared__ int counters[NY];

            // if (group>=0&&lid==0) printf("group  %d  -\n", group);
            for (int col = lid; col < NY; col += dim)
                counters[col] = 0;
            __syncthreads ();
            const int row = group / NY;
            const int bucketEdges = min ((int) sourceIndexes[group], (int) maxOut);
            const int loops = (bucketEdges + dim - 1) / dim;
            for (int loop = 0; loop < loops; loop++)
            {
                int col;
                int counter = 0;
                const int edgeIndex = loop * dim + lid;
                if (edgeIndex < bucketEdges)
                {
                    const int index = group * maxOut + edgeIndex;
                    EdgeOut edge = __ldg (&source[index]);
                    if (!null (edge))
                    {
                        u32 node1 = endpoint (sipkeys, edge, 0);
                        col = (node1 >> ZBITS) & YMASK;
                        counter = min ((int) atomicAdd (counters + col, 1), (int) (FLUSHB2 - 1));
                        tmp[col][counter] = edge;
                    }
                }
                __syncthreads ();
                if (counter == FLUSHB - 1)
                {
                    int localIdx = min (FLUSHB2, counters[col]);
                    int newCount = localIdx % FLUSHB;
                    int nflush = localIdx - newCount;
                    u32 grp = row * NY + col;
                    int cnt = min ((int) atomicAdd (destinationIndexes + grp, nflush), (int) (maxOut - nflush));
#pragma unroll
                    for(int i = 0; i < FLUSHB; i+= TMPPERLL4){
                        destination[((u64) grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *) (&tmp[col][i]);
                    }
                    for (int i = FLUSHB; i < nflush; i += TMPPERLL4)
                        destination[((u64) grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *) (&tmp[col][i]);
                    for (int t = 0; t < newCount; t++)
                    {
                        tmp[col][t] = tmp[col][t + nflush];
                    }
                    counters[col] = newCount;
                }
                __syncthreads ();
            }
            EdgeOut zero = make_Edge (0, tmp[0][0], 0, 0);
            for (int col = lid; col < NY; col += dim)
            {
                int localIdx = min (FLUSHB2, counters[col]);
                u32 grp = row * NY + col;
                for (int j = localIdx; j % TMPPERLL4; j++)
                    tmp[col][j] = zero;

                if (localIdx > 0)
                {
                    int tmpl = (localIdx + TMPPERLL4 - 1) / TMPPERLL4 * TMPPERLL4;
                    int cnt = min ((int) atomicAdd (destinationIndexes + grp, tmpl), (int) (maxOut - tmpl));
                    for (int i = 0; i < localIdx; i += TMPPERLL4)
                    {
                        destination[((u64) grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *) (&tmp[col][i]);
                    }
                }
            }
        }


#ifndef PART_BITS
// #bits used to partition edge set processing to save shared memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif

const u32 PART_MASK = (1 << PART_BITS) - 1; // 1
const u32 NONPART_BITS = ZBITS - PART_BITS; // ZBITS
const u32 NONPART_MASK = (1 << NONPART_BITS) - 1; // 1 << ZBITS
const int BITMAPBYTES = (NZ >> PART_BITS) / 4; // NZ / 8

template<int maxIn, int maxOut>
__global__ void Round(const int round, const int part, const siphash_keys &sipkeys, uint2 * __restrict__ src, uint2 * __restrict__ dst, u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int BITMAPWORDS = BITMAPBYTES / sizeof(u32);

  extern __shared__ u32 ebitmap[];

  for (int i = lid; i < BITMAPWORDS; i += dim)
    ebitmap[i] = 0;
  __syncthreads();
  int edgesInBucket = min(srcIdx[group], maxIn);
  // if (!group && !lid) printf("round %d size  %d\n", round, edgesInBucket);
  int loops = (edgesInBucket + dim-1) / dim;

  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
      uint2 edge = __ldg(&src[index]);
      if (null(edge)) continue;
      u32 z = endpoint(sipkeys, edge, round&1) & ZMASK;
      if ((z >> NONPART_BITS) == part) {
        bitmapset(ebitmap, z & NONPART_MASK);
      }
    }
  }
  __syncthreads();
  edgesInBucket = min(srcIdx[group], maxIn);
  loops = (edgesInBucket + dim-1) / dim;
  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn * group + lindex;
      uint2 edge = __ldg(&src[index]);
      if (null(edge)) continue;
      u32 node0 = endpoint(sipkeys, edge, round&1);
      u32 z = node0 & ZMASK;
      if ((z >> NONPART_BITS) == part && bitmaptest(ebitmap, z)) {
        u32 node1 = endpoint(sipkeys, edge, (round&1)^1);
        const int bucket = node1 >> ZBITS;
        const int bktIdx = min(atomicAdd(dstIdx + bucket, 1), maxOut - 1);
        dst[bucket * maxOut + bktIdx] = (round&1) ? make_uint2(node1, node0) : make_uint2(node0, node1);
        //dst[bucket * maxOut + bktIdx] = (round&1) ? make_Edge(edge, *dst, node1, node0) : make_Edge(edge, *dst, node0, node1);
      }
    }
  }
}

template<int maxIn0, int maxIn1, int maxOut>
__global__ void Round2(const int round, const int part, const siphash_keys &sipkeys, uint2 * __restrict__ src, uint2 * __restrict__ dst, u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
  const int group = blockIdx.x;
  const int dim = blockDim.x;
  const int lid = threadIdx.x;
  const int BITMAPWORDS = BITMAPBYTES / sizeof(u32);

  extern __shared__ u32 ebitmap[];

  for (int i = lid; i < BITMAPWORDS; i += dim)
    ebitmap[i] = 0;
  __syncthreads();

  int edgesInBucket = min(srcIdx[group], maxIn0);
  // if (!group && !lid) printf("round %d size  %d\n", round, edgesInBucket);
  int loops = (edgesInBucket + dim-1) / dim;
  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn0 * group + lindex;
      uint2 edge = __ldg(&src[index]);
      if (null(edge)) continue;
      u32 z = endpoint(sipkeys, edge, round&1) & ZMASK;
      if ((z >> NONPART_BITS) == part) {
        bitmapset(ebitmap, z & NONPART_MASK);
      }
    }
  }
  edgesInBucket = min(srcIdx[NX2 + group], maxIn1);
  // if (!group && !lid) printf("round %d size  %d\n", round, edgesInBucket);
  loops = (edgesInBucket + dim-1) / dim;
  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn1 * group + lindex;
      uint2 edge = __ldg(&src[NX2*maxIn0 + index]);
      if (null(edge)) continue;
      u32 z = endpoint(sipkeys, edge, round&1) & ZMASK;
      if ((z >> NONPART_BITS) == part) {
        bitmapset(ebitmap, z & NONPART_MASK);
      }
    }
  }
  __syncthreads();

  edgesInBucket = min(srcIdx[group], maxIn0);
  loops = (edgesInBucket + dim-1) / dim;
  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn0 * group + lindex;
      uint2 edge = __ldg(&src[index]);
      if (null(edge)) continue;
      u32 node0 = endpoint(sipkeys, edge, round&1);
      u32 z = node0 & ZMASK;
      if ((z >> NONPART_BITS) == part && bitmaptest(ebitmap, z)) {
        u32 node1 = endpoint(sipkeys, edge, (round&1)^1);
        const int bucket = node1 >> ZBITS;
        const int bktIdx = min(atomicAdd(dstIdx + bucket, 1), maxOut - 1);
        dst[bucket * maxOut + bktIdx] = (round&1) ? make_uint2(node1, node0) : make_uint2(node0, node1);
        //dst[bucket * maxOut + bktIdx] = (round&1) ? make_Edge(edge, *dst, node1, node0) : make_Edge(edge, *dst, node0, node1);
      }
    }
  }
  edgesInBucket = min(srcIdx[NX2 + group], maxIn1);
  loops = (edgesInBucket + dim-1) / dim;
  for (int loop = 0; loop < loops; loop++) {
    const int lindex = loop * dim + lid;
    if (lindex < edgesInBucket) {
      const int index = maxIn1 * group + lindex;
      uint2 edge = __ldg(&src[NX2*maxIn0 + index]);
      if (null(edge)) continue;
      u32 node0 = endpoint(sipkeys, edge, round&1);
      u32 z = node0 & ZMASK;
      if ((z >> NONPART_BITS) == part && bitmaptest(ebitmap, z)) {
        u32 node1 = endpoint(sipkeys, edge, (round&1)^1);
        const int bucket = node1 >> ZBITS;
        const int bktIdx = min(atomicAdd(dstIdx + bucket, 1), maxOut - 1);
        dst[bucket * maxOut + bktIdx] = (round&1) ? make_uint2(node1, node0) : make_uint2(node0, node1);
            //make_Edge(edge, *dst, node1, node0) : make_Edge(edge, *dst, node0, node1);
      }
    }
  }
}
    template < int maxIn > __global__ void Tail (const uint2 * source, uint2 * destination, const u32 * sourceIndexes, u32 * destinationIndexes)
    {
        const int lid = threadIdx.x;
        const int group = blockIdx.x;
        int myEdges = sourceIndexes[group];
        __shared__ int destIdx;

        if (lid == 0)
            destIdx = atomicAdd (destinationIndexes, myEdges);

        __syncthreads ();
        if(lid < myEdges)
            destination[destIdx + lid] = source[group * maxIn + lid];
    }

    edgetrimmer::edgetrimmer (const trimparams _tp, u32 _deviceId, int _selected)
    {
        selected = _selected;
        indexesSize = NX * NY * sizeof (u32);
        tp = _tp;

        cudaSetDevice (_deviceId);
        this->deviceId = deviceId;
        checkCudaErrors (cudaMalloc ((void **) &dipkeys, sizeof (siphash_keys)));
        checkCudaErrors (cudaMalloc ((void **) &dipkeys2, sizeof (siphash_keys)));
        checkCudaErrors(cudaMalloc((void**)&uvnodes, indexesSize));


        for (int i = 0; i < NB + 1; i++)
        {
            checkCudaErrors (cudaMalloc ((void **) &indexesE[i], indexesSize));
        }
        //        checkCudaErrors(cudaMalloc((void**)&indexesE2, indexesSize));

        sizeA = ROW_EDGES_A * NX * (selected == 0 && tp.expand > 0 ? sizeof (u32) : sizeof (uint2));
        sizeB = ROW_EDGES_B * NX * (selected == 0 && tp.expand > 1 ? sizeof (u32) : sizeof (uint2));

        const size_t nonoverlap = sizeB * NRB1 / NX;
        const size_t bufferSize = sizeA + nonoverlap;
        assert(bufferSize - sizeB >= sizeB / (tp.expand==2 ? 1 : 2)); // ensure enough space for Round 1, / 2 is for 0.296 / 0.632 without expansion
        checkCudaErrors(cudaMalloc((void**)&bufferA, bufferSize));
        bufferAB = bufferA + nonoverlap;
        bufferB  = bufferA + bufferSize - sizeB;
        assert((NA & (NA-1)) == 0); // ensure NA is a 2 power
        assert(NA * NEPS_B * NRB1 >= NEPS_A * NX); // ensure disjoint source dest in SeedB
        assert(sizeA / NA <= nonoverlap); // equivalent to above
        assert(bufferA + sizeA * NRB2 / NX <= bufferB); // ensure disjoint source dest in 2nd phase of round 0
        assert(bufferA + sizeA == bufferB + sizeB * NRB2 / NX); // ensure alignment of overlap
        cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
        int maxbytes = 0x10000; // 64 KB
        cudaFuncSetAttribute(Round<EDGES_A, EDGES_B*NRB1/NX>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        cudaFuncSetAttribute(Round<EDGES_A, EDGES_B*NRB2/NX>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        cudaFuncSetAttribute(Round2<EDGES_B*NRB2/NX, EDGES_B*NRB1/NX, EDGES_B/2>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        cudaFuncSetAttribute(Round<EDGES_B/2, EDGES_A/4>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        cudaFuncSetAttribute(Round<EDGES_A/4, EDGES_B/4>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        cudaFuncSetAttribute(Round<EDGES_B/4, EDGES_B/4>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    }
    u64 edgetrimmer::globalbytes () const
    {
        return (sizeA+sizeB*NRB1/NX) + (1+NB) * indexesSize + sizeof(siphash_keys) + PROOFSIZE * 2*sizeof(u32) + sizeof(edgetrimmer);
    }
    edgetrimmer::~edgetrimmer ()
    {
        cudaSetDevice (deviceId);
        cudaFree (bufferA);
        for (int i = 0; i < NB + 1; i++)
        {
            cudaFree (indexesE[i]);
        }
        cudaFree (dipkeys);
        cudaDeviceReset ();
    }

    int com (const void *a, const void *b)
    {
        uint2 va = *(uint2 *) a;
        uint2 vb = *(uint2 *) b;
        if (va.x == vb.y)
            return va.y - vb.y;
        else
            return va.x - vb.x;
    }

    void saveFile (uint2 * v, int n, char *filename)
    {
        qsort (v, n, sizeof (uint2), com);
        FILE *fp = fopen (filename, "w");
        for (int i = 0; i < n; i++)
        {
            fprintf (fp, "%d,%d\n", v[i].x, v[i].y);
        }
        fclose (fp);
    }
    u32 edgetrimmer::trim (uint32_t device)
    {
        cudaSetDevice (device);

#ifdef TIMER
        cudaEvent_t start, stop;
        checkCudaErrors (cudaEventCreate (&start));
        checkCudaErrors (cudaEventCreate (&stop));
#endif

        cudaMemset (indexesE[1], 0, indexesSize);
        cudaMemcpy (dipkeys, &sipkeys, sizeof (sipkeys), cudaMemcpyHostToDevice);

//        checkCudaErrors (cudaDeviceSynchronize ());

#ifdef TIMER
        float durationA, durationB;
        cudaEventRecord (start, NULL);
#endif

        if (selected == 0)
        {
            if (tp.expand == 0)
                Cuckoo_SeedA < EDGES_A, uint2 ><<< tp.genA.blocks, tp.genA.tpb >>> (*dipkeys, (ulonglong4 *) bufferAB, (int *) indexesE[1]);
            else
                Cuckoo_SeedA < EDGES_A, u32 ><<< tp.genA.blocks, tp.genA.tpb >>> (*dipkeys, (ulonglong4 *) bufferAB, (int *) indexesE[1]);
        }
        else
            Cuckaroo_SeedA < EDGES_A ><<< tp.genA.blocks, tp.genA.tpb >>> (*dipkeys, (ulonglong4 *) bufferAB, (int *) indexesE[1]);

//        checkCudaErrors (cudaDeviceSynchronize ());

#ifdef TIMER
        cudaEventRecord (stop, NULL);
        cudaEventSynchronize (stop);
        cudaEventElapsedTime (&durationA, start, stop);

        cudaEventRecord (start, NULL);
#endif

        cudaMemset (indexesE[0], 0, indexesSize);
        size_t qA = sizeA / NA;
        size_t qE = NX2 / NA;
        for (u32 i = 0; i < NA; i++)
        {
            if (selected != 0 || tp.expand == 0)
            {
                SeedB < EDGES_A, uint2 ><<< tp.genB.blocks / NA, tp.genB.tpb >>> (*dipkeys, (const uint2 *) (bufferAB + i * qA), (ulonglong4 *) (bufferA + i * qA), indexesE[1] + i * qE,
                        indexesE[0] + i * qE);
            }
            else
            {
                SeedB < EDGES_A, u32 ><<< tp.genB.blocks / NA, tp.genB.tpb >>> (*dipkeys, (const u32 *) (bufferAB + i * qA), (ulonglong4 *) (bufferA + i * qA), indexesE[1] + i * qE,
                        indexesE[0] + i * qE);
            }
        }

#ifdef INDEX_DEBUG
        cudaMemcpy (hostA, indexesE[0], NX * NY * sizeof (u32), cudaMemcpyDeviceToHost);
        fprintf (stderr, "Index Number: %zu\n", hostA[0]);
#endif

#ifdef TIMER
        cudaEventRecord (stop, NULL);
        cudaEventSynchronize (stop);
        cudaEventElapsedTime (&durationB, start, stop);
        fprintf (stderr, "Seeding completed in %.2f + %.2f ms\n", durationA, durationB);

        cudaEventRecord (start, NULL);
#endif

  
    cudaMemset(indexesE[2], 0, indexesSize);

    qA = sizeA * NRB2 / NX;
    qE = NY * NRB2;
    for (u32 part = 0; part <= PART_MASK; part++) {
        Round<EDGES_A, EDGES_B*NRB1/NX><<<tp.trim.blocks*NRB1/NX, tp.trim.tpb, BITMAPBYTES>>>(0, part, *dipkeys, (uint2*)(bufferA+qA), (uint2*)(bufferA+sizeA), indexesE[0]+qE, indexesE[2]); // to .632
    }

    cudaMemset(indexesE[1], 0, indexesSize);

    for (u32 part = 0; part <= PART_MASK; part++) {
        Round<EDGES_A, EDGES_B*NRB2/NX><<<tp.trim.blocks*NRB2/NX, tp.trim.tpb, BITMAPBYTES>>>(0, part, *dipkeys, (uint2*)bufferA, (uint2*)bufferB, indexesE[0], indexesE[1]); // to .632
    }

    cudaMemset(indexesE[0], 0, indexesSize);

    for (u32 part = 0; part <= PART_MASK; part++) {
        Round2<EDGES_B*NRB2/NX, EDGES_B*NRB1/NX, EDGES_B/2><<<tp.trim.blocks, tp.trim.tpb, BITMAPBYTES>>>(1, part, *dipkeys, (uint2*)bufferB, (uint2*)bufferA, indexesE[1], indexesE[0]); // to .296
    }

    cudaMemset(indexesE[1], 0, indexesSize);

    for (u32 part = 0; part <= PART_MASK; part++) {
        Round<EDGES_B/2, EDGES_A/4><<<tp.trim.blocks, tp.trim.tpb, BITMAPBYTES>>>(2, part, *dipkeys, (uint2 *)bufferA, (uint2 *)bufferB, indexesE[0], indexesE[1]); // to .176
    }

    cudaMemset(indexesE[0], 0, indexesSize);

    for (u32 part = 0; part <= PART_MASK; part++) {
      Round<EDGES_A/4, EDGES_B/4><<<tp.trim.blocks, tp.trim.tpb, BITMAPBYTES>>>(3, part, *dipkeys, (uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]); // to .117
    }
  
//    cudaDeviceSynchronize();
  
    for (int round = 4; round < tp.ntrims; round += 2) {
      cudaMemset(indexesE[1], 0, indexesSize);
      for (u32 part = 0; part <= PART_MASK; part++) {
        Round<EDGES_B/4, EDGES_B/4><<<tp.trim.blocks, tp.trim.tpb, BITMAPBYTES>>>(round  , part, *dipkeys, (uint2 *)bufferA, (uint2 *)bufferB, indexesE[0], indexesE[1]);
      }
      cudaMemset(indexesE[0], 0, indexesSize);
      for (u32 part = 0; part <= PART_MASK; part++) {
        Round<EDGES_B/4, EDGES_B/4><<<tp.trim.blocks, tp.trim.tpb, BITMAPBYTES>>>(round+1, part, *dipkeys, (uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]);
      }
    }
    
    cudaMemset(indexesE[1], 0, indexesSize);
    cudaDeviceSynchronize();
  
    Tail<EDGES_B/4><<<tp.tail.blocks, tp.tail.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, (const u32 *)indexesE[0], (u32 *)indexesE[1]);
//    cudaMemcpy(&nedges, indexesE[1], sizeof(u32), cudaMemcpyDeviceToHost);
//    cudaDeviceSynchronize();
    bool ready = false;
    while(1){
        usleep(1000);
        ready = cudaSuccess == cudaStreamQuery(0);
        if(ready){
            cudaMemcpy(&nedges, indexesE[1], sizeof(u32), cudaMemcpyDeviceToHost);
            break;
        }
    }
    return nedges;
    }

};
