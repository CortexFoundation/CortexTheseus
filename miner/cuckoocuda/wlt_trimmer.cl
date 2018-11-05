//#include "miner/cuckoocuda/src/trimmer_cl.h"


#define TROMP_SEEDA
#define TROMP_SEEDB
#define TROMP_ROUND
#define TROMP_TAIL

#define TIMER


// proof-of-work parameters
#ifndef EDGEBITS
// the main parameter is the 2-log of the graph size,
// which is the size in bits of the node identifiers
#define EDGEBITS 24
#endif
#ifndef PROOFSIZE
// the next most important parameter is the (even) length
// of the cycle to be found. a minimum of 12 is recommended
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

// number of edges
#define NEDGES ((node_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK ((edge_t)NEDGES - 1)

#ifndef XBITS
#define XBITS ((EDGEBITS-16)/2)
#endif

#define NODEBITS (EDGEBITS + 1)
#define NNODES ((node_t)1 << NODEBITS)
#define NODEMASK (NNODES - 1)

#define IDXSHIFT 10
#define CUCKOO_SIZE (NNODES >> IDXSHIFT)
#define CUCKOO_MASK (CUCKOO_SIZE - 1)
// number of (least significant) key bits that survives leftshift by NODEBITS
#define KEYBITS (64-NODEBITS)
#define KEYMASK ((1L << KEYBITS) - 1)
#define MAXDRIFT (1L << (KEYBITS - IDXSHIFT))

#define MAXEDGES  0x1000000
#define NX        (1 << XBITS)
#define NX2       ((NX) * (NX))
#define XMASK     ((NX) - 1)
#define X2MASK    ((NX2) - 1)
#define YBITS      XBITS
#define NY         (1 << (YBITS))
#define YZBITS  ((EDGEBITS) - (XBITS))
#define NYZ       (1 << (YZBITS))
#define ZBITS     ((YZBITS) - (YBITS))
#define NZ        (1 << (ZBITS))
#define COUNTERWORDS  ((NZ) / 16)

#define EPS_A 133/128
#define EPS_B 85/128

#define ROW_EDGES_A ((NYZ) * (EPS_A))
#define ROW_EDGES_B ((NYZ) * (EPS_B))

#define EDGES_A ((ROW_EDGES_A) / (NX))
#define EDGES_B ((ROW_EDGES_B) / (NX))

#define DUCK_A_EDGES (EDGES_A)
#define DUCK_A_EDGES_NX (DUCK_A_EDGES * NX)
#define DUCK_B_EDGES (EDGES_B)
#define DUCK_B_EDGES_NX (DUCK_B_EDGES * NX)

#ifndef FLUSHA // should perhaps be in trimparams and passed as template parameter
#define FLUSHA 16
#endif

#ifndef FLUSHB
#define FLUSHB 8
#endif

typedef struct {
    long k0;
    long k1;
    long k2;
    long k3;
} siphash_keys;

#define U8TO64_LE(p) ((p))
#define ROTL(x,b) (long)( ((x) << (b)) | ( (x) >> (64 - (b))) )
  #define SIPROUND \
    do { \
      v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
      v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
      v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
      v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
      v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
    } while(0)
 
#define FLUSHB2 (2 * FLUSHB)
#define FLUSHA2  (2 * FLUSHA)

inline ulong4 make_ulong4(ulong r1, ulong r2, ulong r3, ulong r4){
    return (ulong4)(r1,r2,r3,r4);
}
inline uint2 make_uint2(uint a, uint b){
    return (uint2)(a, b);
}

inline ulong4 Pack4edges(const uint2 e1, const  uint2 e2, const  uint2 e3, const  uint2 e4)
{
    ulong r1 = (((ulong)e1.y << 32) | ((ulong)e1.x));
    ulong r2 = (((ulong)e2.y << 32) | ((ulong)e2.x));
    ulong r3 = (((ulong)e3.y << 32) | ((ulong)e3.x));
    ulong r4 = (((ulong)e4.y << 32) | ((ulong)e4.x));
    return make_ulong4(r1, r2, r3, r4);
}


inline ulong4 uint2_to_ulong4(uint2 v0, uint2 v1, uint2 v2, uint2 v3){
//	return Pack4edges(v0, v1, v2, v3);
	
	ulong l0 = ((ulong)v0.x << 32 | v0.y);
	ulong l1 = ((ulong)v1.x << 32 | v1.y);
	ulong l2 = ((ulong)v2.x << 32 | v2.y);
	ulong l3 = ((ulong)v3.x << 32 | v3.y);
	return (ulong4)(l0, l1, l2, l3);
}
inline node_t dipnode(__constant const siphash_keys* keys, edge_t nce, uint uorv) {
  long nonce = 2*nce + uorv; 
  long v0 = (*keys).k0, v1 = (*keys).k1, v2 = (*keys).k2, v3 = (*keys).k3^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}

inline uint endpoint(__constant const siphash_keys* sipkeys, uint nonce, int uorv) {
	return dipnode(sipkeys, nonce, uorv);
}

inline uint endpoint2(__constant const siphash_keys* sipkeys, uint2 nodes, int uorv) {
	return uorv ? nodes.y : nodes.x;
}

inline uint2 make_Edge_by_node(const uint nonce, const uint2 dummy, const uint node0, const uint node1) {
	return make_uint2(node0, node1);
}

inline uint2 make_Edge_by_edge(const uint2 edge, const uint2 dummy, const uint node0, const uint node1) {
	return edge;
}

inline uint make_Edge_by_nonce(const uint nonce, const uint dummy, const uint node0, const uint node1) {
	return nonce;
}


inline  void Increase2bCounter(__local uint *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  uint mask = 1 << bit;

  uint old = atomic_or(ecounters + word, mask) & mask;
  if (old)
    atomic_or(ecounters + word + NZ/32, mask);
}

inline  bool Read2bCounter(__local uint *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  uint mask = 1 << bit;

  return (ecounters[word + NZ/32] & mask) != 0;
}

//    __constant__ uint2 e0 = {0,0};

    inline ulong4 Pack8(const uint e0, const uint e1, const uint e2, const uint e3, const uint e4, const uint e5, const uint e6, const uint e7) {
        return make_ulong4((long)e0<<32|e1, (long)e2<<32|e3, (long)e4<<32|e5, (long)e6<<32|e7);
    }

    inline bool null(uint nonce) {
        return (nonce == 0);
    }

    inline bool null2(uint2 nodes) {
        return (nodes.x == 0 && nodes.y == 0);
    }

// ===== Above =======

//__constant__ uint2 recoveredges[PROOFSIZE];

#ifdef TROMP_ROUND
__kernel void Recovery(__constant const siphash_keys *sipkeys, __global ulong4 *buffer, __global int *indexes, __global uint2* recoveredges) {
  const int gid = get_global_id(0);//blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = get_local_id(0);//threadIdx.x;
  const int nthreads = get_global_size(0);//blockDim.x * gridDim.x;
  const int loops = NEDGES / nthreads;
  __local uint nonces[PROOFSIZE];

  if (lid < PROOFSIZE) nonces[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 0; i < loops; i++) {
    long nonce = gid * loops + i;
    long u = dipnode(sipkeys, nonce, 0);
    long v = dipnode(sipkeys, nonce, 1);
    for (int i = 0; i < PROOFSIZE; i++) {
      if (recoveredges[i].x == u && recoveredges[i].y == v)
        nonces[i] = nonce;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid < PROOFSIZE) {
    if (nonces[lid] > 0)
      indexes[lid] = nonces[lid];
  }
}

#else
__kernel void Recovery(__global const siphash_keys *sipkeys, __global ulong4 *buffer, __global int *indexes, __global uint2* recoveredges) {
  const int gid = get_global_id(0);//blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = get_local_id(0);//threadIdx.x;
  const int nthreads = get_global_size(0);//blockDim.x * gridDim.x;
  const int loops = NEDGES / nthreads;
  __local uint nonces[PROOFSIZE];

  if (lid < PROOFSIZE) nonces[lid] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 0; i < loops; i++) {
    long nonce = gid * loops + i;
    long u = dipnode(sipkeys, nonce, 0);
    long v = dipnode(sipkeys, nonce, 1);
    for (int i = 0; i < PROOFSIZE; i++) {
      if (recoveredges[i].x == u && recoveredges[i].y == v)
        nonces[i] = nonce;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < PROOFSIZE) {
    if (nonces[lid] > 0)
      indexes[lid] = nonces[lid];
  }
}
#endif


//    template<int maxOut, typename EdgeOut>
        __kernel void SeedA(__constant const siphash_keys *sipkeys, __global ulong4 * __restrict__ buffer, __global int * __restrict__ indexes, int maxOut, int bufferAB_offset) {
            const int group = get_group_id(0);//blockIdx.x;
            const int dim = get_local_size(0);//blockDim.x;
            const int lid = get_local_id(0);//threadIdx.x;
            const int gid = group * dim + lid;
            const int nthreads = get_global_size(0);//gridDim.x * dim;
            //const int FLUSHA2 = 2*FLUSHA;
           if(gid == 0){
	   	printf("EDGEBITS = %d, PROOFSIZE = %d\n", EDGEBITS, PROOFSIZE);
	   } 
			__local uint2 tmp[NX][FLUSHA2]; // needs to be ulong4 aligned
            const int TMPPERLL4 = sizeof(ulong4) / sizeof(uint2);
            __local int counters[NX];

            for (int row = lid; row < NX; row += dim)
                counters[row] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);

            const int col = group % NX;
            const int loops = NEDGES / nthreads;
            for (int i = 0; i < loops; i++) {
                uint nonce = gid * loops + i;
                uint node1, node0 = dipnode(sipkeys, (long)nonce, 0);
                if (sizeof(uint2) == sizeof(uint2))
                    node1 = dipnode(sipkeys, (long)nonce, 1);
                int row = node0 & XMASK;
                int counter = min((int)atomic_add(counters + row, 1), (int)(FLUSHA2-1));
                tmp[row][counter] = make_Edge_by_node(nonce, tmp[0][0], node0, node1);
                barrier(CLK_LOCAL_MEM_FENCE);
                if (counter == FLUSHA-1) {
                    int localIdx = min(FLUSHA2, counters[row]);
                    int newCount = localIdx % FLUSHA;
                    int nflush = localIdx - newCount;
                    int cnt = min((int)atomic_add(indexes + row * NX + col, nflush), (int)(maxOut - nflush));
                    for (int i = 0; i < nflush; i += TMPPERLL4)
                        //buffer[((long)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4] = *(ulong4 *)(&tmp[row][i]);
                        buffer[((long)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4] = uint2_to_ulong4(tmp[row][i], tmp[row][i+1], tmp[row][i+2], tmp[row][i+3]);
                    for (int t = 0; t < newCount; t++) {
                        tmp[row][t] = tmp[row][t + nflush];
                    }
                    counters[row] = newCount;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            uint2 zero = make_Edge_by_node(0, tmp[0][0], 0, 0);
            for (int row = lid; row < NX; row += dim) {
                int localIdx = min(FLUSHA2, counters[row]);
                for (int j = localIdx; j % TMPPERLL4; j++)
                    tmp[row][j] = zero;
                for (int i = 0; i < localIdx; i += TMPPERLL4) {
                    int cnt = min((int)atomic_add(indexes + row * NX + col, TMPPERLL4), (int)(maxOut - TMPPERLL4));
                    //buffer[((long)(row * NX + col) * maxOut + cnt) / TMPPERLL4] = *(ulong4 *)(&tmp[row][i]);
                    buffer[((long)(row * NX + col) * maxOut + cnt) / TMPPERLL4] = uint2_to_ulong4(tmp[row][i], tmp[row][i+1], tmp[row][i+2], tmp[row][i+3]);
                }
            }
        }

    __kernel void Seed2A(__constant const siphash_keys *sipkeys, __global ulong4 * __restrict__ buffer, __global int * __restrict__ indexes, int bufferAB_offset) {
        const int group = get_group_id(0);//blockIdx.x;
        const int dim = get_local_size(0);//blockDim.x;
        const int lid = get_local_id(0);//threadIdx.x;
        const int nthreads = get_num_groups(0) * (dim-NX);//gridDim.x * (dim - NX);
        //const int FLUSHA2 = 2 * FLUSHA;
        /* const int FLUSHA2 = 16; */

        __local uint2 tmp[NX][FLUSHA2]; // needs to be ulong4 aligned
        __local int counters[NX];

        if (lid < NX) counters[lid] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        const int col = group % NX;
        const int loops = (NEDGES) / nthreads;
        const int gid = group * (dim - NX) + (lid - NX);
        for (int i = 0; i < loops; i++) {
            int bucket, counter;
            uint node0, node1;
            if (lid >= NX) {
                uint nonce = gid * loops + i;

                node0 = dipnode(sipkeys, nonce, 0);
                node1 = dipnode(sipkeys, nonce, 1);
                bucket = node0 & XMASK;
                counter = min((int)atomic_add(counters + bucket, 1), (int)(FLUSHA2-1));
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid >= NX)
                tmp[bucket][counter] = make_uint2(node0, node1);

            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid >= NX) continue;

            int localIdx = min(FLUSHA2, counters[lid]);
            if (localIdx >= 4) {
                int newCount = (localIdx % 4);
                int nflush = localIdx - newCount;
                counters[lid] = newCount;

                int cnt = min((int)atomic_add(indexes + lid * NX + col, nflush), (int)(DUCK_A_EDGES - nflush));

                for (int l = 0; l < nflush; l += 4)
                    //buffer[((long)(lid * NX + col) * DUCK_A_EDGES + cnt + l) / 4] = *(ulong4 *)(&tmp[lid][l]);
                    buffer[bufferAB_offset + ((long)(lid * NX + col) * DUCK_A_EDGES + cnt + l) / 4] = uint2_to_ulong4(tmp[lid][l], tmp[lid][l+1], tmp[lid][l+2], tmp[lid][l+3]);

                for (int t = 0; t < newCount; t++) {
                    tmp[lid][t] = tmp[lid][t + nflush];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= NX) return ;

        int localIdx = counters[lid];
        if (localIdx > 0) {
            int cnt = min((int)atomic_add(indexes + lid * NX + col, 4), (int)(DUCK_A_EDGES - 4));
            uint2 zero = make_uint2(0, 0);
            for (int j = localIdx; j % 4 != 0; j++)
                tmp[lid][j] = zero;
            //buffer[((long)(lid * NX + col) * DUCK_A_EDGES + cnt) / 4] = *(ulong4 *)(&tmp[lid][0]);
            buffer[bufferAB_offset + ((long)(lid * NX + col) * DUCK_A_EDGES + cnt) / 4] = uint2_to_ulong4(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
        }
    }

//    template<int maxOut, typename EdgeOut>
        __kernel void SeedB(__constant const siphash_keys *sipkeys, __global const uint2 * __restrict__ source, __global ulong4 * __restrict__ destination, __global const int * __restrict__ sourceIndexes, __global int * __restrict__ destinationIndexes, const int maxOut, const uint halfA, const uint halfE, int bufferAB_offset){ 
            const int group = get_group_id(0);//blockIdx.x;
            const int dim = get_local_size(0);//blockDim.x;
            const int lid = get_local_id(0);//threadIdx.x;
            //const int FLUSHB2 = 2 * FLUSHB;

			const int gid = get_global_id(0);

            __local uint2 tmp[NX][FLUSHB2];
            const int TMPPERLL4 = sizeof(ulong4) / sizeof(uint2);
            __local int counters[NX];
            
			// if (group>=0&&lid==0) printf("group  %d  -\n", group);
            for (int col = lid; col < NX; col += dim)
                counters[col] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);

            const int row = group / NX;
            const int bucketEdges = min((int)sourceIndexes[group + halfE], (int)maxOut);
            const int loops = (bucketEdges + dim-1) / dim;
            
			for (int loop = 0; loop < loops; loop++) {
                int col; int counter = 0;
                const int edgeIndex = loop * dim + lid;
            
				if (edgeIndex < bucketEdges) {
                    const int index = group * maxOut + edgeIndex;
                    //uiint2 edge = __ldg(&source[index + halfA]);
                    uint2 edge = source[bufferAB_offset/sizeof(uint2) + index + halfA/sizeof(uint2)];
                    if (null2(edge)) continue;
                    uint node1 = endpoint2(sipkeys, edge, 0);
                    col = (node1 >> XBITS) & XMASK;
                    counter = min((int)atomic_add(counters+col, 1), (int)(FLUSHB2-1));
                    tmp[col][counter] = edge;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
				if (counter == FLUSHB-1) {
                    int localIdx = min(FLUSHB2, counters[col]);
                    int newCount = localIdx % FLUSHB;
                    int nflush = localIdx - newCount;
                    int cnt = min((int)atomic_add(destinationIndexes + row * NX + col + halfE, nflush), (int)(maxOut - nflush));
                    for (int i = 0; i < nflush; i += TMPPERLL4)
                        //destination[((long)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4 + halfA] = *(ulong4 *)(&tmp[col][i]);
                        destination[((long)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4 + halfA/sizeof(ulong4)] = uint2_to_ulong4(tmp[col][i], tmp[col][i+1], tmp[col][i+2], tmp[col][i+3]);
                    for (int t = 0; t < newCount; t++) {
                        tmp[col][t] = tmp[col][t + nflush];
                    }
                    counters[col] = newCount;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            uint2 zero = make_Edge_by_node(0, tmp[0][0], 0, 0);
            for (int col = lid; col < NX; col += dim) {
                int localIdx = min(FLUSHB2, counters[col]);
//				printf("%d\n", localIdx);
                for (int j = localIdx; j % TMPPERLL4; j++)
                    tmp[col][j] = zero;
                for (int i = 0; i < localIdx; i += TMPPERLL4) {
                    int cnt = min((int)atomic_add(destinationIndexes + row * NX + col + halfE, TMPPERLL4), (int)(maxOut - TMPPERLL4));
                    //destination[((long)(row * NX + col) * maxOut + cnt) / TMPPERLL4 + halfA] = *(ulong4 *)(&tmp[col][i]);
                    destination[((long)(row * NX + col) * maxOut + cnt) / TMPPERLL4 + halfA/sizeof(ulong4)] = uint2_to_ulong4(tmp[col][i], tmp[col][i+1], tmp[col][i+2], tmp[col][i+3]);
                }
            }
        }

/* #define BKTGRAN NX / 2 */
    __kernel void Seed2B(__global const uint2 * __restrict__ source, __global ulong4 * __restrict__ destination, __global const int * __restrict__ sourceIndexes, __global int * __restrict__ destinationIndexes, const int halfA, const int halfE, int bufferAB_offset) {
        const int group = get_group_id(0);//blockIdx.x;
        const int dim = get_local_size(0);//blockDim.x;
        const int lid = get_local_id(0);//threadIdx.x;
        //const int FLUSHB2 = 2 * FLUSHB;

        __local uint2 tmp[NX][FLUSHB2];
        /* const int TMPPERLL4 = sizeof(ulong4) / sizeof(uint2); */
        __local int counters[NX];

        counters[lid] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);
        /* const int myBucket = group / NX; */
        /* const int microBlockNo = group % NX; */
        /* const int bucketEdges = min(sourceIndexes[myBucket * NX + halfE], (int)(DUCK_A_EDGES_64)); */
        /* const int microBlockEdgesCount = (DUCK_A_EDGES_64 / BKTGRAN); */
        /* const int loops = (microBlockEdgesCount / NX); */

        const int row = group / NX;
        const int bucketEdges = min((int)sourceIndexes[group + halfE], (int)DUCK_A_EDGES);
        const int loops = (bucketEdges + dim-1) / dim;
        for (int loop = 0; loop < loops; loop++) {
            const int edgeIndex = loop * dim + lid;
            if (edgeIndex < bucketEdges) {
                const int index = group * DUCK_A_EDGES + edgeIndex;
                uint2 edge = source[bufferAB_offset + index + halfA];
                if (edge.x == 0 && edge.y == 0) continue;
                int bucket = (edge.x >> XBITS) & XMASK;

                barrier(CLK_LOCAL_MEM_FENCE);
                int counter = min((int)atomic_add(counters + bucket, 1), (int)(FLUSHB2-1));
                tmp[bucket][counter] = edge;

                barrier(CLK_LOCAL_MEM_FENCE);
                int localIdx = min(FLUSHB2, counters[lid]);
                if (localIdx >= 4) {
                    int newCount = (localIdx % 4);
                    int nflush = localIdx - newCount;
                    counters[lid] = newCount;

                    int cnt = min((int)atomic_add(destinationIndexes + row * NX + lid + halfE, nflush), (int)(DUCK_A_EDGES - nflush));

                    for (int i = 0; i < nflush; i += 4)
                        //destination[((row * NX + lid) * DUCK_A_EDGES + cnt + i) / 4] = *(ulong4 *)(&tmp[lid][i]);
                        destination[((row * NX + lid) * DUCK_A_EDGES + cnt + i) / 4 + halfA] = uint2_to_ulong4(tmp[lid][i], tmp[lid][i+1], tmp[lid][i+2], tmp[lid][i+3]);

                    for (int t = 0; t < newCount; t++) {
                        tmp[lid][t] = tmp[lid][t + nflush];
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        int localIdx = min(FLUSHB2, counters[lid]);
        if (localIdx > 0) {
            uint2 zero = make_uint2(0, 0);
            int cnt = min((int)atomic_add(destinationIndexes + row * NX + lid + halfE, 4), (int)(DUCK_A_EDGES - 4));
            for (int j = localIdx; j % 4 != 0; j++)
                tmp[lid][j] = zero;
            //destination[((long)(row * NX + lid) * DUCK_A_EDGES + cnt) / 4 + halfA] = *(ulong4 *)(&tmp[lid][0]);
            destination[((long)(row * NX + lid) * DUCK_A_EDGES + cnt) / 4 + halfA] = uint2_to_ulong4(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
        }
    }

//    template<int maxIn, typename EdgeIn, int maxOut, typename EdgeOut>
        __kernel void Round(const int round, __constant const siphash_keys *sipkeys, __global const uint2 * __restrict__ source, __global uint2 * __restrict__ destination, __global const int * __restrict__ sourceIndexes, __global int * __restrict__ destinationIndexes, const int maxIn, const int maxOut, int src_offset, int dest_offset) {
            const int group = get_group_id(0);//blockIdx.x;
            const int dim = get_local_size(0);//blockDim.x;
            const int lid = get_local_id(0);//threadIdx.x;
            //const static int COUNTERWORDS = NZ / 16; // 16 2-bit counters per 32-bit word

            __local uint ecounters[COUNTERWORDS];
            for (int i = lid; i < COUNTERWORDS; i += dim)
                ecounters[i] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            const int edgesInBucket = min(sourceIndexes[group], maxIn);
            const int loops = (edgesInBucket + dim-1) / dim;

            for (int loop = 0; loop < loops; loop++) {
                const int lindex = loop * dim + lid;
                if (lindex < edgesInBucket) {
                    const int index = maxIn * group + lindex;
                    //uint2 edge = __ldg(&source[index]);
                    uint2 edge = source[index];
                    if (null2(edge)) continue;
                    uint node = endpoint2(sipkeys, edge, round&1);
                    Increase2bCounter(ecounters, node >> (2*XBITS));
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int loop = 0; loop < loops; loop++) {
                const int lindex = loop * dim + lid;
                if (lindex < edgesInBucket) {
                    const int index = maxIn * group + lindex;
                    //uint2 edge = __ldg(&source[index]);
                    uint2 edge = source[index];
                    if (null2(edge)) continue;
                    uint node0 = endpoint2(sipkeys, edge, round&1);
                    if (Read2bCounter(ecounters, node0 >> (2*XBITS))) {
                        uint node1 = endpoint2(sipkeys, edge, (round&1)^1);
                        const int bucket = node1 & X2MASK;
                        const int bktIdx = min(atomic_add(destinationIndexes + bucket, 1), maxOut - 1);
                        destination[bucket * maxOut + bktIdx] = (round&1) ? make_Edge_by_edge(edge, *destination, node1, node0)
                            : make_Edge_by_edge(edge, *destination, node0, node1);
                    }
                }
            }
            // if (group==0&&lid==0) printf("round %d cnt(0,0) %d\n", round, sourceIndexes[0]);
        }

//    template<int maxIn, int maxOut>
        __kernel void Round2(__global const uint2 * __restrict__ source, __global uint2 * __restrict__ destination, __global const int * __restrict__ sourceIndexes, __global int * __restrict__ destinationIndexes, const int maxIn, const int maxOut, int src_offset, int dest_offset) {
            const int group = get_group_id(0);//blockIdx.x;
            const int dim = get_local_size(0);//blockDim.x;
            const int lid = get_local_id(0);//threadIdx.x;
           // const static int COUNTERWORDS = NZ / 16; // 16 2-bit counters per 32-bit word

            const int edgesInBucket = min(sourceIndexes[group], maxIn);
            const int loops = (edgesInBucket + dim-1) / dim;

            __local uint ecounters[COUNTERWORDS];
            for (int i = lid; i < COUNTERWORDS; i += dim)
                ecounters[i] = 0;

            barrier(CLK_LOCAL_MEM_FENCE);
            for (int loop = 0; loop < loops; loop++) {
                const int lindex = loop * dim + lid;

                if (lindex < edgesInBucket) {
                    const int index = maxIn * group + lindex;
                    uint2 edge = source[index + src_offset/sizeof(uint2)];
                    if (edge.x == 0 && edge.y == 0) continue;
                    Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> (2*XBITS));
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            for (int loop = 0; loop < loops; loop++) {
                const int lindex = loop * dim + lid;

                if (lindex < edgesInBucket) {
                    const int index = maxIn * group + lindex;
                    uint2 edge = source[index + src_offset/sizeof(uint2)];
                    if (edge.x == 0 && edge.y == 0) continue;

                    if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> (2*XBITS))) {
                        const int bucket = edge.y & X2MASK;
                        const int bktIdx = min(atomic_add(destinationIndexes + bucket, 1), maxOut - 1);
                        destination[bucket * maxOut + bktIdx + dest_offset/sizeof(uint)] = make_uint2(edge.y, edge.x);
                    }
                }
            }
        }

//    template<int maxIn>
        __kernel void Tail(__global const uint2 *source, __global uint2 *destination, __global const int *sourceIndexes, __global int *destinationIndexes, const int maxIn, int dest_offset) {
      const int lid = get_local_id(0);//threadIdx.x;
      const int group = get_group_id(0);//blockIdx.x;
      const int dim = get_local_size(0);//blockDim.x;
      int myEdges = sourceIndexes[group];
      __local int destIdx;

      if (lid == 0)
        destIdx = atomic_add(destinationIndexes, myEdges);

      barrier(CLK_LOCAL_MEM_FENCE);
      for (int i = lid; i < myEdges; i += dim)
        destination[destIdx + lid + dest_offset/sizeof(uint2)] = source[group * maxIn + lid];
    }

//    template<int maxIn>
        __kernel void Tail2(__global const uint2 *source, __global uint2 *destination, __global const int *sourceIndexes, __global int *destinationIndexes, const int maxIn, int dest_offset) {
      const int lid = get_local_id(0);//threadIdx.x;
      const int group = get_group_id(0);//blockIdx.x;
      const int dim = get_local_size(0);//blockDim.x;
      int myEdges = sourceIndexes[group];
      __local int destIdx;

      if (lid == 0)
        destIdx = atomic_add(destinationIndexes, myEdges);

      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid < myEdges) {
        destination[destIdx + lid + dest_offset/sizeof(uint2)] = source[group * maxIn + lid];
      }
    }

 
