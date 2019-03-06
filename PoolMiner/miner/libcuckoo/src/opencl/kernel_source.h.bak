#ifndef WLT_TRIMMER_H
#define WLT_TRIMMER_H
#include <string>
static const std::string kernel_source = R"(
#ifndef EDGEBITS
#define EDGEBITS 25
#endif
#ifndef PROOFSIZE
#define PROOFSIZE 4
#endif

#if EDGEBITS > 32
typedef ulong edge_t;
#else
typedef uint edge_t;
#endif
#if EDGEBITS > 31
typedef ulong node_t;
#else
typedef uint node_t;
#endif

#define NEDGES ((node_t)1 << EDGEBITS)
#define EDGEMASK ((edge_t)NEDGES - 1)

#ifndef XBITS
#define XBITS ((EDGEBITS-16)/2)
#endif

#define NX        (1 << (XBITS))
#define NX2       ((NX) * (NX))
#define XMASK     ((NX) - 1)
#define X2MASK    ((NX2) - 1)
#define YBITS      XBITS
#define YZBITS  ((EDGEBITS) - (XBITS))
#define ZBITS     ((YZBITS) - (YBITS))
#define NZ        (1 << (ZBITS))
#define COUNTERWORDS  ((NZ) / 16)
#define ZMASK     (NZ - 1)

#ifndef FLUSHA			// should perhaps be in trimparams and passed as template parameter
#define FLUSHA 16
#endif

#ifndef FLUSHB
#define FLUSHB 8
#endif

#ifndef EDGE_BLOCK_BITS
#define EDGE_BLOCK_BITS 6
#endif
#define EDGE_BLOCK_SIZE (1 << EDGE_BLOCK_BITS)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)


typedef struct {
    ulong k0;
    ulong k1;
    ulong k2;
    ulong k3;
} siphash_keys;

#define U8TO64_LE(p) ((p))
#define ROTL(x,b) (ulong)( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
    do { \
      v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
      v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
      v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
      v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
      v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
    } while(0)

#define FLUSHB2 (2 * (FLUSHB))
#define FLUSHA2  (2 * (FLUSHA))



#ifndef EXPAND
#define EXPAND 0
#endif

#if (EXPAND == 0)
 #define EdgeOut uint2
 #define EdgeIn uint2
#else
 #define EdgeIn uint
#endif

inline ulong4
make_ulong4(ulong r1, ulong r2, ulong r3, ulong r4)
{
    return (ulong4) (r1, r2, r3, r4);
}

inline uint2
make_uint2(uint a, uint b)
{
    return (uint2) (a, b);
}

inline ulong4
Pack4edges(const uint2 e1, const uint2 e2, const uint2 e3, const uint2 e4)
{
    ulong r1 = (((ulong) e1.y << 32) | ((ulong) e1.x));
    ulong r2 = (((ulong) e2.y << 32) | ((ulong) e2.x));
    ulong r3 = (((ulong) e3.y << 32) | ((ulong) e3.x));
    ulong r4 = (((ulong) e4.y << 32) | ((ulong) e4.x));
    return make_ulong4(r1, r2, r3, r4);
}

inline ulong4
uint2_to_ulong4(uint2 v0, uint2 v1, uint2 v2, uint2 v3)
{
    return Pack4edges(v0, v1, v2, v3);
}

inline node_t
dipnode(__constant const siphash_keys * keys, edge_t nce, uint uorv)
{
    ulong nonce = 2 * nce + uorv;
    ulong v0 = (*keys).k0, v1 = (*keys).k1, v2 = (*keys).k2, v3 = (*keys).k3 ^ nonce;
    SIPROUND;
    SIPROUND;
    v0 ^= nonce;
    v2 ^= 0xff;
    SIPROUND;
    SIPROUND;
    SIPROUND;
    SIPROUND;
    return (v0 ^ v1 ^ v2 ^ v3) & EDGEMASK;
}

inline uint
endpoint(__constant const siphash_keys * sipkeys, uint nonce, int uorv)
{
    return dipnode(sipkeys, nonce, uorv);
}

inline uint
endpoint2(__constant const siphash_keys * sipkeys, uint2 nodes, int uorv)
{
    return uorv ? nodes.y : nodes.x;
}

inline uint2
make_Edge_by_node(const uint nonce, const uint2 dummy, const uint node0, const uint node1)
{
    return make_uint2(node0, node1);
}

inline uint2
make_Edge_by_edge(const uint2 edge, const uint2 dummy, const uint node0, const uint node1)
{
    return edge;
}

inline uint
make_Edge_by_nonce(const uint nonce, const uint dummy, const uint node0, const uint node1)
{
    return nonce;
}

inline void
Increase2bCounter(__local uint * ecounters, const int bucket)
{
    int word = bucket >> 5;
    unsigned char bit = bucket & 0x1F;
    uint mask = 1 << bit;

    uint old = atomic_or(ecounters + word, mask) & mask;
    if (old)
	atomic_or(ecounters + word + NZ / 32, mask);
}

inline bool
Read2bCounter(__local uint * ecounters, const int bucket)
{
    int word = bucket >> 5;
    unsigned char bit = bucket & 0x1F;
    uint mask = 1 << bit;

    return (ecounters[word + NZ / 32] & mask) != 0;
}

inline ulong4
Pack8(const uint e0, const uint e1, const uint e2, const uint e3, const uint e4, const uint e5, const uint e6, const uint e7)
{
    return make_ulong4((long) e0 << 32 | e1, (long) e2 << 32 | e3, (long) e4 << 32 | e5, (long) e6 << 32 | e7);
}

inline bool
null(uint nonce)
{
    return (nonce == 0);
}

inline bool
null2(uint2 nodes)
{
    return (nodes.x == 0 && nodes.y == 0);
}


inline ulong dipblock(__constant const siphash_keys *key, const edge_t edge, ulong *buf) {
  //diphash_state shs(keys);
  siphash_keys keys = *key;
  ulong v0 = keys.k0, v1 = keys.k1, v2 = keys.k2, v3 = keys.k3;

  edge_t edge0 = edge & ~EDGE_BLOCK_MASK;
  uint i;
  for (i=0; i < EDGE_BLOCK_MASK; i++) {
    //shs.hash24(edge0 + i);
	  edge_t nonce = edge0 + i;
	v3^=nonce;
	SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;	
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;

//    buf[i] = shs.xor_lanes();
	buf[i] = (v0 ^ v1) ^ (v2  ^ v3);
  }
//  shs.hash24(edge0 + i);
	edge_t nonce = edge0 + i;
    v3^=nonce;
	SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;

//    buf[i] = shs.xor_lanes();
	buf[i] = 0;
  //return shs.xor_lanes();
	return (v0 ^ v1) ^ (v2  ^ v3);
}

__kernel void Cuckaroo_Recovery(__constant const siphash_keys *sipkeys,__global int *indexes, __constant uint2* recoveredges) {
  const int gid = get_global_id(0);//blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = get_local_id(0);//threadIdx.x;
  const int nthreads = get_global_size(0);//blockDim.x * gridDim.x;
  const int loops = NEDGES / nthreads;
  __local uint nonces[PROOFSIZE];
  ulong buf[EDGE_BLOCK_SIZE];

  if (lid < PROOFSIZE) nonces[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
    uint nonce0 = gid * loops + blk;
    const ulong last = dipblock(sipkeys, nonce0, buf);
    for (int i = 0; i < EDGE_BLOCK_SIZE; i++) {
      ulong edge = buf[i] ^ last;
      uint u = edge & EDGEMASK;
      uint v = (edge >> 32) & EDGEMASK;
      for (int p = 0; p < PROOFSIZE; p++) {
        if (recoveredges[p].x == u && recoveredges[p].y == v)
          nonces[p] = nonce0 + i;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < PROOFSIZE) {
    if (nonces[lid] > 0)
      indexes[lid] = nonces[lid];
  }
}

__kernel void
Cuckoo_Recovery(__constant const siphash_keys * sipkeys, __global int *indexes, __constant uint2 * recoveredges)
{
    const int gid = get_global_id(0);	//blockDim.x * blockIdx.x + threadIdx.x;
    const int lid = get_local_id(0);	//threadIdx.x;
    const int nthreads = get_global_size(0);	//blockDim.x * gridDim.x;
    const int loops = NEDGES / nthreads;
    __local uint nonces[PROOFSIZE];

    if (lid < PROOFSIZE)
	nonces[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < loops; i++)
    {
	ulong nonce = gid * loops + i;
	ulong u = dipnode(sipkeys, nonce, 0);
	ulong v = dipnode(sipkeys, nonce, 1);
	for (int i = 0; i < PROOFSIZE; i++)
	{
	    if (recoveredges[i].x == u && recoveredges[i].y == v)
		nonces[i] = nonce;
	}
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < PROOFSIZE)
    {
	if (nonces[lid] > 0)
	    indexes[lid] = nonces[lid];
    }
}

__kernel void 
Cuckaroo_SeedA(__constant const siphash_keys* sipkeys,
		__global uchar * __restrict__ cbuffer,
		__global uint * __restrict__ indexes, 
		const int maxOut, const uint offset, const uint idx_offset) {
  const int group = get_group_id(0);//blockIdx.x;
  const int dim = get_local_size(0);//blockDim.x;
  const int lid = get_local_id(0);//threadIdx.x;
  const int gid = group * dim + lid;
  const int nthreads = get_global_size(0);//gridDim.x * dim;
  //const int FLUSHA2 = 2*FLUSHA;
  
  __global uint2* buffer =(__global uint2*)(cbuffer + offset);

  __local uint2 tmp[NX][FLUSHA2]; // needs to be ulonglong4 aligned
  const int TMPPERLL4 = sizeof(ulong4) / sizeof(uint2);
  __local int counters[NX];
  ulong buf[EDGE_BLOCK_SIZE];

  for (int row = lid; row < NX; row += dim)
    counters[row] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

 
//  const uint tmp_offset = offset / sizeof(uint2);
  const int col = group % NX;
  const int loops = NEDGES / nthreads; // assuming THREADS_HAVE_EDGES checked
  for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
   uint nonce0 = gid * loops + blk;
    const ulong last = dipblock(sipkeys, nonce0, buf);
    for (uint e = 0; e < EDGE_BLOCK_SIZE; e++) {
      ulong edge = buf[e] ^ last;
      uint node0 = edge & EDGEMASK;
      uint node1 = (edge >> 32) & EDGEMASK;
      int row = node0 >> YZBITS;
      int counter = min((int)atomic_add(counters + row, 1), (int)(FLUSHA2-1)); // assuming ROWS_LIMIT_LOSSES checked
      tmp[row][counter] = make_uint2(node0, node1);
      barrier(CLK_LOCAL_MEM_FENCE);
      if (counter == FLUSHA-1) {
        int localIdx = min(FLUSHA2, counters[row]);
        int newCount = localIdx % FLUSHA;
        int nflush = localIdx - newCount;
		uint grp = row*NX + col;
        int cnt = min((int)atomic_add(indexes + grp + idx_offset, nflush), (int)(maxOut - nflush));
        for (int i = 0; i < nflush; i += 1){
		//	uint2 t[4];
		//	t[0] = tmp[row][i]; t[1]=tmp[row][i+1]; t[2] = tmp[row][i+2]; t[3] = tmp[row][i+3];
			buffer[((ulong)grp * maxOut + cnt + i)] = tmp[row][i];//*(ulong4*)&t;//*(__local ulong4*)(&tmp[row][i]);
		}
        for (int t = 0; t < newCount; t++) {
          tmp[row][t] = tmp[row][t + nflush];
        }
        counters[row] = newCount;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  uint2 zero = make_uint2(0, 0);
  for (int row = lid; row < NX; row += dim) {
    int localIdx = min(FLUSHA2, counters[row]);
    uint grp = row * NX + col;
    for (int j = localIdx; j % TMPPERLL4; j++)
      tmp[row][j] = zero;
    
	if(localIdx > 0){
	int cnt = min((int)atomic_add(indexes + grp + idx_offset, localIdx), (int)(maxOut - localIdx));
    for (int i = 0; i < localIdx; i += 1) {
//      int cnt = min((int)atomicAdd(indexes + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
//		uint2 t[4];
//		t[0] = tmp[row][i]; t[1]=tmp[row][i+1]; t[2] = tmp[row][i+2]; t[3] = tmp[row][i+3];
		buffer[((ulong)grp * maxOut + cnt + i)] = tmp[row][i];
    }
	}
  }
}

__kernel void
Cuckoo_SeedA(__constant const siphash_keys * sipkeys, __global EdgeIn * __restrict__ buffer, __global uint *__restrict__ indexes, const uint maxOut, const uint offset, const uint idx_offset)
{
    const int group = get_group_id(0);
    const int dim = get_local_size(0);
    const int lid = get_local_id(0);
    const int gid = group * dim + lid;
    const int nthreads = get_global_size(0);

    __local EdgeIn tmp[NX][FLUSHA2];
    __local int counters[NX];

    for (int row = lid; row < NX; row += dim)
		counters[row] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint tmp_offset = offset / sizeof(EdgeIn);
    const int col = group % NX;
    const int loops = NEDGES / nthreads;
    for (int i = 0; i < loops; i++)
    {
		uint nonce = gid * loops + i;
		uint node1, node0 = dipnode(sipkeys, (ulong) nonce, 0);
		if (sizeof (EdgeIn) == sizeof (uint2))
			node1 = dipnode(sipkeys, (ulong) nonce, 1);
		int row = node0 >> YZBITS;
		int counter = min((int) atomic_add(counters + row, 1), (int) (FLUSHA2 - 1));
#if (EXPAND == 0)
		tmp[row][counter] = make_Edge_by_node(nonce, tmp[0][0], node0, node1);
#else 
		tmp[row][counter] = make_Edge_by_nonce(nonce, tmp[0][0], node0, node1);
#endif
		barrier(CLK_LOCAL_MEM_FENCE);
		if (counter == FLUSHA - 1)
		{
			int localIdx = min(FLUSHA2, counters[row]);
			int newCount = localIdx % FLUSHA;
			int nflush = localIdx - newCount;
			int cnt = min((int) atomic_add(indexes + row * NX + col + idx_offset, nflush),
				  (int) (maxOut - nflush));
			for (int i = 0; i < nflush; i += 1)
				buffer[tmp_offset + ((ulong) (row * NX + col) * maxOut + cnt + i)] = tmp[row][i];
			for (int t = 0; t < newCount; t++)
			{
				tmp[row][t] = tmp[row][t + nflush];
			}
			counters[row] = newCount;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int row = lid; row < NX; row += dim)
    {
		int localIdx = min(FLUSHA2, counters[row]);
		if(localIdx > 0){
		int cnt = min((int) atomic_add(indexes + row * NX + col + idx_offset, localIdx),
			  (int) (maxOut - localIdx));
		for (int i = 0; i < localIdx; i += 1)
		{
			buffer[tmp_offset + ((ulong) (row * NX + col) * maxOut + cnt + i)] = tmp[row][i];
		}
		}
    }
}

__kernel void
SeedB(__constant const siphash_keys * sipkeys,
      __global uchar * __restrict__ tsrc,
      __global uchar * __restrict__ tdst,
      __global const uint *__restrict__ srcIdx, 
	  __global uint *__restrict__ dstIdx, 
	  const int maxOut, const uint halfA, 
	  const uint halfE, uint offset,
	  const uint srcIdx_offset, const uint dstIdx_offset)
{
    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    const int lid = get_local_id(0);	//threadIdx.x;

    const int gid = get_global_id(0);

	__global EdgeIn* src = (__global EdgeIn*)(tsrc + halfA + offset);
	__global uint2 *dst = (__global uint2*)(tdst + halfA);
    __local EdgeIn tmp[NX][FLUSHB2];

	const int TMPPERLL4 = sizeof(ulong4) / sizeof(EdgeIn);
    __local int counters[NX];

    for (int col = lid; col < NX; col += dim)
		counters[col] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int row = group / NX;
    const int bucketEdges = min((int) srcIdx[group + halfE + srcIdx_offset], (int) maxOut);
    const int loops = (bucketEdges + dim - 1) / dim;

    for (int loop = 0; loop < loops; loop++)
    {
		int col;
		int counter = 0;
		const int edgeIndex = loop * dim + lid;

		if (edgeIndex < bucketEdges)
		{
			const int index = group * maxOut + edgeIndex;
			EdgeIn edge = src[index];
#if (EXPAND == 0)
			if (null2(edge)) continue;
			uint node1 = endpoint2(sipkeys, edge, 0);
#else
			if (null(edge)) continue;
			uint node1 = endpoint(sipkeys, edge, 0);
#endif
			col = (node1 >> ZBITS) & XMASK;
			counter = min((int) atomic_add(counters + col, 1), (int) (FLUSHB2 - 1));
			tmp[col][counter] = edge;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if (counter == FLUSHB - 1)
		{
			int localIdx = min(FLUSHB2, counters[col]);
			int newCount = localIdx % FLUSHB;
			int nflush = localIdx - newCount;
			int cnt = min((int) atomic_add(dstIdx + row * NX + col + halfE + dstIdx_offset,
						   nflush), (int) (maxOut - nflush));
			for (int i = 0; i < nflush; i += 1)
			{
				/*
				ulong t[4];
				t[0] = tmp[col][i].x | ((ulong)tmp[col][i].y << 32);
				t[1] = tmp[col][i+1].x | ((ulong)tmp[col][i+1].y << 32);
				t[2] = tmp[col][i+2].x | ((ulong)tmp[col][i+2].y << 32);
				t[3] = tmp[col][i+3].x | ((ulong)tmp[col][i+3].y << 32);
				*/
				dst[((ulong) (row * NX + col) * maxOut + cnt + i)] = tmp[col][i];//(ulong4)(t[0],t[1],t[2],t[3]);
			}
			for (int t = 0; t < newCount; t++)
			{
				tmp[col][t] = tmp[col][t + nflush];
			}
			counters[col] = newCount;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }
	uint2 zero = make_uint2(0,0);
    for (int col = lid; col < NX; col += dim)
    {
		int localIdx = min(FLUSHB2, counters[col]);
		for(int j = localIdx; j % TMPPERLL4; j++)
			tmp[col][j] = zero;

		if(localIdx > 0){
			int cnt = min((int) atomic_add(dstIdx + row * NX + col + halfE + dstIdx_offset, localIdx), (int) (maxOut - localIdx));
			for (int i = 0; i < localIdx; i += 1)
			{
//				uint2 t[4];
//				t[0] = tmp[col][i]; t[1]=tmp[col][i+1]; t[2] = tmp[col][i+2]; t[3] = tmp[col][i+3];
				dst[((ulong) (row * NX + col) * maxOut + cnt + i)] = tmp[col][i];//*(ulong4*)&t;
			}
		}
    }
}

__kernel void
Round(const int round, __constant const siphash_keys * sipkeys,
      __global const uchar * __restrict__ tsrc,
      __global uchar * __restrict__ tdst,
	  __global const uint *__restrict__ srcIdx, 
	  __global uint *__restrict__ dstIdx, 
	  const uint maxIn, const uint maxOut, 
	  const uint src_offset, const uint dest_offset,
	  const uint srcIdx_offset, const uint dstIdx_offset, const uint NP)
{

	__global const uint2* src = (__global const uint2*)(tsrc + src_offset);
	__global uint2 * dst = (__global uint2*)(tdst + dest_offset);

    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    const int lid = get_local_id(0);	//threadIdx.x;
    __local uint ecounters[COUNTERWORDS];
    for (int i = lid; i < COUNTERWORDS; i += dim)
		ecounters[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
		const int edgesInBucket = min(srcIdx[group + srcIdx_offset], maxIn);
		const int loops = (edgesInBucket + dim - 1) / dim;
		for (int loop = 0; loop < loops; loop++)
		{
			const int lindex = loop * dim + lid;
			if (lindex < edgesInBucket)
			{
				const int index = maxIn * group + lindex;
				uint2 edge = src[index];
				if (null2(edge))
					continue;
				uint node = endpoint2(sipkeys, edge, round & 1);
				Increase2bCounter(ecounters, node & ZMASK);
			}
		}
	  }

    barrier(CLK_LOCAL_MEM_FENCE);

  src -= NP * NX2 * maxIn; srcIdx -= NP * NX2;
  for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
    const int edgesInBucket = min(srcIdx[group + srcIdx_offset], maxIn);
    const int loops = (edgesInBucket + dim - 1) / dim;
    for (int loop = 0; loop < loops; loop++)
    {
		const int lindex = loop * dim + lid;
		if (lindex < edgesInBucket)
		{
			const int index = maxIn * group + lindex;
			uint2 edge = src[index];
			if (null2(edge))
				continue;
			uint node0 = endpoint2(sipkeys, edge, round & 1);

			if (Read2bCounter(ecounters, node0 & ZMASK))
			{
				uint node1 = endpoint2(sipkeys, edge, (round & 1) ^ 1);
				const int bucket = node1 >> ZBITS;
				const int bktIdx = min(atomic_add(dstIdx + bucket + dstIdx_offset, 1), maxOut - 1);
				dst[bucket * maxOut + bktIdx] =  (round&1) ? make_uint2(node1, node0) : make_uint2(node0, node1);

			}
		}
    }
  }
}

__kernel void
Round_uint_uint2(const int round, __constant const siphash_keys * sipkeys,
      __global const uint * __restrict__ src,
      __global uint2 * __restrict__ dst,
	  __global const uint *__restrict__ srcIdx, 
	  __global uint *__restrict__ dstIdx, 
	  const uint maxIn, const uint maxOut, 
	  const uint src_offset, const uint dest_offset,
	  const uint srcIdx_offset, const uint dstIdx_offset, const uint NP)
{
    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    const int lid = get_local_id(0);	//threadIdx.x;
    __local uint ecounters[COUNTERWORDS];
    for (int i = lid; i < COUNTERWORDS; i += dim)
		ecounters[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
    const int edgesInBucket = min(srcIdx[group + srcIdx_offset], maxIn);
    const int loops = (edgesInBucket + dim - 1) / dim;

    for (int loop = 0; loop < loops; loop++)
    {
		const int lindex = loop * dim + lid;
		if (lindex < edgesInBucket)
		{
			const int index = maxIn * group + lindex;
			uint edge = src[index + src_offset / sizeof(uint)];
			if (null(edge))
				continue;
			uint node = endpoint(sipkeys, edge, round & 1);
			Increase2bCounter(ecounters, node & ZMASK);
		}
    }
  }
    barrier(CLK_LOCAL_MEM_FENCE);
  src -= NP * NX2 * maxIn; srcIdx -= NP * NX2;
  for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
    const int edgesInBucket = min(srcIdx[group + srcIdx_offset], maxIn);
    const int loops = (edgesInBucket + dim - 1) / dim;
    for (int loop = 0; loop < loops; loop++)
    {
		const int lindex = loop * dim + lid;
		if (lindex < edgesInBucket)
		{
			const int index = maxIn * group + lindex;
			uint edge = src[index + src_offset/sizeof(uint)];
			if (null(edge))
				continue;
			
			uint node0 = endpoint(sipkeys, edge, round & 1);
			if (Read2bCounter(ecounters, node0 & ZMASK))
			{
				uint node1 = endpoint(sipkeys, edge, (round & 1) ^ 1);
				const int bucket = node1 >> ZBITS;
				const int bktIdx = min(atomic_add(dstIdx + (bucket + dstIdx_offset), 1), maxOut - 1);
				dst[bucket * maxOut + bktIdx + dest_offset/sizeof(uint2)] = (round&1) ? make_uint2(node1, node0) : make_uint2(node0, node1);
			}
		}
    }
  }
}

__kernel void
Round_uint_uint(const int round, __constant const siphash_keys * sipkeys,
      __global const uint * __restrict__ src,
      __global uint * __restrict__ dst,
	  __global const uint *__restrict__ srcIdx,
	  __global uint *__restrict__ dstIdx, 
	  const uint maxIn, const uint maxOut, 
	  const uint src_offset, const uint dest_offset,
	  const uint srcIdx_offset, const uint dstIdx_offset, const uint NP)
{
    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    const int lid = get_local_id(0);	//threadIdx.x;
    __local uint ecounters[COUNTERWORDS];
    for (int i = lid; i < COUNTERWORDS; i += dim)
		ecounters[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
    const int edgesInBucket = min(srcIdx[group + srcIdx_offset], maxIn);
    const int loops = (edgesInBucket + dim - 1) / dim;

    for (int loop = 0; loop < loops; loop++)
    {
		const int lindex = loop * dim + lid;
		if (lindex < edgesInBucket)
		{
			const int index = maxIn * group + lindex;
			uint edge = src[index + src_offset / sizeof(uint)];
			if (null(edge))
				continue;
			
			uint node = endpoint(sipkeys, edge, round & 1);
			Increase2bCounter(ecounters, node & ZMASK);
		}
    }
  }
    barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
    const int edgesInBucket = min(srcIdx[group + srcIdx_offset], maxIn);
    const int loops = (edgesInBucket + dim - 1) / dim;
    for (int loop = 0; loop < loops; loop++)
    {
		const int lindex = loop * dim + lid;
		if (lindex < edgesInBucket)
		{
			const int index = maxIn * group + lindex;
			uint edge = src[index + src_offset/sizeof(uint)];
			if (null(edge))
				continue;

			uint node0 = endpoint(sipkeys, edge, round & 1);
			if (Read2bCounter(ecounters, node0 & ZMASK))
			{
				uint node1 = endpoint(sipkeys, edge, (round & 1) ^ 1);
				const int bucket = node1 >> ZBITS;
				const int bktIdx = min(atomic_add(dstIdx + bucket, 1), maxOut - 1);
				dst[bucket * maxOut + bktIdx + dest_offset/sizeof(uint)] = edge;
			}
		}
    }
  }
}


__kernel void
Tail(__global const uint2 * source, __global uchar * destination, __global const uint *srcIdx, __global uint *dstIdx, const int maxIn, uint offset, const uint srcIdx_offset, const uint dstIdx_offset)
{
	__global uint2* dst = (__global uint2*)(destination + offset);
    const int lid = get_local_id(0);	//threadIdx.x;
    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    int myEdges = srcIdx[group + srcIdx_offset];
    __local int destIdx;

    if (lid == 0)
	destIdx = atomic_add(dstIdx + dstIdx_offset, myEdges);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < myEdges; i += dim)
	dst[destIdx + i] = source[group * maxIn + i];
}

)";

inline std::string get_kernel_source ()
{
	return kernel_source;
}
#endif
