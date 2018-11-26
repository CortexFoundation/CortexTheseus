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

#ifndef FLUSHA			// should perhaps be in trimparams and passed as template parameter
#define FLUSHA 16
#endif

#ifndef FLUSHB
#define FLUSHB 8
#endif

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

__kernel void
Recovery(__constant const siphash_keys * sipkeys, __global int *indexes, __constant uint2 * recoveredges)
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
SeedA(__constant const siphash_keys * sipkeys, __global uint2 * __restrict__ buffer, __global int *__restrict__ indexes, int maxOut, uint offset)
{
    const int group = get_group_id(0);
    const int dim = get_local_size(0);
    const int lid = get_local_id(0);
    const int gid = group * dim + lid;
    const int nthreads = get_global_size(0);
    __local uint2 tmp[NX][FLUSHA2];
    __local int counters[NX];

    for (int row = lid; row < NX; row += dim)
	counters[row] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint tmp_offset = offset / sizeof(uint2);
    const int col = group % NX;
    const int loops = NEDGES / nthreads;
    for (int i = 0; i < loops; i++)
    {
	uint nonce = gid * loops + i;
	uint node1, node0 = dipnode(sipkeys, (ulong) nonce, 0);
	if (sizeof (uint2) == sizeof (uint2))
	    node1 = dipnode(sipkeys, (ulong) nonce, 1);
	int row = node0 & XMASK;
	int counter = min((int) atomic_add(counters + row, 1), (int) (FLUSHA2 - 1));
	tmp[row][counter] = make_Edge_by_node(nonce, tmp[0][0], node0, node1);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (counter == FLUSHA - 1)
	{
	    int localIdx = min(FLUSHA2, counters[row]);
	    int newCount = localIdx % FLUSHA;
	    int nflush = localIdx - newCount;
	    int cnt = min((int) atomic_add(indexes + row * NX + col, nflush),
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
	int cnt = min((int) atomic_add(indexes + row * NX + col, 1),
		  (int) (maxOut - 1));
	for (int i = 0; i < localIdx; i += 1)
	{
	    buffer[tmp_offset + ((ulong) (row * NX + col) * maxOut + cnt + i)] = tmp[row][i];
	}
    }
}

__kernel void
SeedB(__constant const siphash_keys * sipkeys,
      __global const uint2 * __restrict__ source,
      __global uint2 * __restrict__ destination,
      __global const int *__restrict__ sourceIndexes, __global int *__restrict__ destinationIndexes, const int maxOut, const uint halfA, const uint halfE, uint offset)
{
    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    const int lid = get_local_id(0);	//threadIdx.x;

    const int gid = get_global_id(0);

    __local uint2 tmp[NX][FLUSHB2];
    __local int counters[NX];

    for (int col = lid; col < NX; col += dim)
	counters[col] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int row = group / NX;
    const int bucketEdges = min((int) sourceIndexes[group + halfE], (int) maxOut);
    const int loops = (bucketEdges + dim - 1) / dim;
    const uint dest_halfA = halfA/sizeof(uint2);
    const uint src_halfA = halfA/sizeof(uint2);
    const uint tmp_offset = offset / sizeof(uint2);

    for (int loop = 0; loop < loops; loop++)
    {
	int col;
	int counter = 0;
	const int edgeIndex = loop * dim + lid;

	if (edgeIndex < bucketEdges)
	{
	    const int index = group * maxOut + edgeIndex;
	    uint2 edge = source[index + src_halfA + tmp_offset];
	    if (null2(edge))
		continue;
	    uint node1 = endpoint2(sipkeys, edge, 0);
	    col = (node1 >> XBITS) & XMASK;
	    counter = min((int) atomic_add(counters + col, 1), (int) (FLUSHB2 - 1));
	    tmp[col][counter] = edge;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (counter == FLUSHB - 1)
	{
	    int localIdx = min(FLUSHB2, counters[col]);
	    int newCount = localIdx % FLUSHB;
	    int nflush = localIdx - newCount;
	    int cnt = min((int) atomic_add(destinationIndexes + row * NX + col + halfE,
					   nflush), (int) (maxOut - nflush));
	    for (int i = 0; i < nflush; i += 1)
	    {
		destination[((ulong) (row * NX + col) * maxOut + cnt +
			     i) + dest_halfA] = tmp[col][i];
	    }
	    for (int t = 0; t < newCount; t++)
	    {
		tmp[col][t] = tmp[col][t + nflush];
	    }
	    counters[col] = newCount;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int col = lid; col < NX; col += dim)
    {
	int localIdx = min(FLUSHB2, counters[col]);
	int cnt = min((int) atomic_add(destinationIndexes + row * NX + col + halfE,
			   1), (int) (maxOut - 1));
	for (int i = 0; i < localIdx; i += 1)
	{
	    destination[((ulong) (row * NX + col) * maxOut + cnt + i) +
			dest_halfA] = tmp[col][i];
	}
    }
}

__kernel void
Round(const int round, __constant const siphash_keys * sipkeys,
      __global const uint2 * __restrict__ source,
      __global uint2 * __restrict__ destination, __global const int *__restrict__ sourceIndexes, __global int *__restrict__ destinationIndexes, const int maxIn, const int maxOut, uint src_offset, uint dest_offset)
{
    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    const int lid = get_local_id(0);	//threadIdx.x;
    __local uint ecounters[COUNTERWORDS];
    for (int i = lid; i < COUNTERWORDS; i += dim)
	ecounters[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    const int edgesInBucket = min(sourceIndexes[group], maxIn);
    const int loops = (edgesInBucket + dim - 1) / dim;

    for (int loop = 0; loop < loops; loop++)
    {
	const int lindex = loop * dim + lid;
	if (lindex < edgesInBucket)
	{
	    const int index = maxIn * group + lindex;
	    uint2 edge = source[index + src_offset / sizeof(uint2)];
	    if (null2(edge))
		continue;
	    uint node = endpoint2(sipkeys, edge, round & 1);
	    Increase2bCounter(ecounters, node >> (2 * XBITS));
	}
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int loop = 0; loop < loops; loop++)
    {
	const int lindex = loop * dim + lid;
	if (lindex < edgesInBucket)
	{
	    const int index = maxIn * group + lindex;
	    uint2 edge = source[index + src_offset/sizeof(uint2)];
	    if (null2(edge))
		continue;
	    uint node0 = endpoint2(sipkeys, edge, round & 1);
	    if (Read2bCounter(ecounters, node0 >> (2 * XBITS)))
	    {
		uint node1 = endpoint2(sipkeys, edge, (round & 1) ^ 1);
		const int bucket = node1 & X2MASK;
		const int bktIdx = min(atomic_add(destinationIndexes + bucket, 1), maxOut - 1);
		destination[bucket * maxOut + bktIdx + dest_offset/sizeof(uint2)] = edge;
	    }
	}
    }
}

__kernel void
Tail(__global const uint2 * source, __global uint2 * destination, __global const int *sourceIndexes, __global int *destinationIndexes, const int maxIn, uint offset)
{
    const int lid = get_local_id(0);	//threadIdx.x;
    const int group = get_group_id(0);	//blockIdx.x;
    const int dim = get_local_size(0);	//blockDim.x;
    int myEdges = sourceIndexes[group];
    __local int destIdx;

    if (lid == 0)
	destIdx = atomic_add(destinationIndexes, myEdges);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < myEdges; i += dim)
	destination[destIdx + i + offset/sizeof(uint2)] = source[group * maxIn + i];
}

)";

inline std::string get_kernel_source(){
	return kernel_source;
}
#endif
