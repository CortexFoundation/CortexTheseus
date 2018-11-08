#include <time.h>

#include "trimmer.h" 

namespace cuckoogpu {

#define DUCK_A_EDGES (EDGES_A)
#define DUCK_A_EDGES_NX (DUCK_A_EDGES * NX)
#define DUCK_B_EDGES (EDGES_B)
#define DUCK_B_EDGES_NX (DUCK_B_EDGES * NX)
#define COUNTERWORDS  ((NZ) / 16)

 
#define FLUSHB2 (2 * (FLUSHB))
#define FLUSHA2  (2 * (FLUSHA))

__device__ ulong4 Pack4edges(const uint2 e1, const  uint2 e2, const  uint2 e3, const  uint2 e4)
{
	ulong r1 = (((ulong)e1.y << 32) | ((ulong)e1.x));
	ulong r2 = (((ulong)e2.y << 32) | ((ulong)e2.x));
	ulong r3 = (((ulong)e3.y << 32) | ((ulong)e3.x));
	ulong r4 = (((ulong)e4.y << 32) | ((ulong)e4.x));
	return make_ulong4(r1, r2, r3, r4);
}

__device__ node_t dipnode(const siphash_keys *keys, edge_t nce, u32 uorv) {
  ulong nonce = 2*nce + uorv;
  ulong v0 = (*keys).k0, v1 = (*keys).k1, v2 = (*keys).k2, v3 = (*keys).k3^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}


__global__ void Recovery(const siphash_keys *sipkeys, ulong4 *buffer, int *indexes, uint2* recoveredges) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  const int loops = NEDGES / nthreads;
  __shared__ u32 nonces[PROOFSIZE];

  if (lid < PROOFSIZE) nonces[lid] = 0;
  __syncthreads();
  for (int i = 0; i < loops; i++) {
	ulong nonce = gid * loops + i;
	ulong u = dipnode(sipkeys, nonce, 0);
	ulong v = dipnode(sipkeys, nonce, 1);
	for (int i = 0; i < PROOFSIZE; i++) {
	  if (recoveredges[i].x == u && recoveredges[i].y == v)
		nonces[i] = nonce;
	}
  }
  __syncthreads();
  if (lid < PROOFSIZE) {
	if (nonces[lid] > 0)
	  indexes[lid] = nonces[lid];
  }
}


__device__ __forceinline__  void Increase2bCounter(u32 *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  u32 mask = 1 << bit;

  u32 old = atomicOr(ecounters + word, mask) & mask;
  if (old)
    atomicOr(ecounters + word + NZ/32, mask);
}

__device__ __forceinline__  bool Read2bCounter(u32 *ecounters, const int bucket) {
  int word = bucket >> 5;
  unsigned char bit = bucket & 0x1F;
  u32 mask = 1 << bit;

  return (ecounters[word + NZ/32] & mask) != 0;
}

 //   __constant__ uint2 e0 = {0,0};

    __device__ __forceinline__ ulong4 Pack8(const u32 e0, const u32 e1, const u32 e2, const u32 e3, const u32 e4, const u32 e5, const u32 e6, const u32 e7) {
        return make_ulong4((ulong)e0<<32|e1, (ulong)e2<<32|e3, (ulong)e4<<32|e5, (ulong)e6<<32|e7);
    }

    __device__ bool null(u32 nonce) {
        return nonce == 0;
    }

    __device__ bool null(uint2 nodes) {
        return nodes.x == 0 && nodes.y == 0;
    }

    __device__ u32 endpoint(const siphash_keys *sipkeys, uint2 nodes, int uorv) {
        return uorv ? nodes.y : nodes.x;
    }

    __device__ uint2 make_Edge_by_node(const u32 nonce, const uint2 dummy, const u32 node0, const u32 node1) {
        return make_uint2(node0, node1);
    }

    __device__ uint2 make_Edge_by_edge(const uint2 edge, const uint2 dummy, const u32 node0, const u32 node1) {
        return edge;
    }

//	template<typename EdgeOut>
		__global__ void SeedA(const siphash_keys *sipkeys, ulong4 * __restrict__ buffer, int * __restrict__ indexes, int maxOut) {
			const int group = blockIdx.x;
			const int dim = blockDim.x;
			const int lid = threadIdx.x;
			const int gid = group * dim + lid;
			const int nthreads = gridDim.x * dim;
			//const int FLUSHA2 = 2*FLUSHA;

			__shared__ uint2 tmp[NX][FLUSHA2]; // needs to be ulong4 aligned
			const int TMPPERLL4 = sizeof(ulong4) / sizeof(uint2);
			__shared__ int counters[NX];

			for (int row = lid; row < NX; row += dim)
				counters[row] = 0;
			__syncthreads();

			const int col = group % NX;
			const int loops = NEDGES / nthreads;
			for (int i = 0; i < loops; i++) {
				u32 nonce = gid * loops + i;
				u32 node1, node0 = dipnode(sipkeys, (ulong)nonce, 0);
				if (sizeof(uint2) == sizeof(uint2))
					node1 = dipnode(sipkeys, (ulong)nonce, 1);
				int row = node0 & XMASK;
				int counter = min((int)atomicAdd(counters + row, 1), (int)(FLUSHA2-1));
				tmp[row][counter] = make_Edge_by_node(nonce, tmp[0][0], node0, node1);
				__syncthreads();
				if (counter == FLUSHA-1) {
					int localIdx = min(FLUSHA2, counters[row]);
					int newCount = localIdx % FLUSHA;
					int nflush = localIdx - newCount;
					int cnt = min((int)atomicAdd(indexes + row * NX + col, nflush), (int)(maxOut - nflush));
					for (int i = 0; i < nflush; i += TMPPERLL4)
						buffer[((ulong)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4] = Pack4edges(tmp[row][i], tmp[row][i+1], tmp[row][i+2], tmp[row][i+3]);//*(ulong4 *)(&tmp[row][i]);
					for (int t = 0; t < newCount; t++) {
						tmp[row][t] = tmp[row][t + nflush];
					}
					counters[row] = newCount;
				}
				__syncthreads();
			}
			uint2 zero = make_Edge_by_node(0, tmp[0][0], 0, 0);
			for (int row = lid; row < NX; row += dim) {
				int localIdx = min(FLUSHA2, counters[row]);
				for (int j = localIdx; j % TMPPERLL4; j++)
					tmp[row][j] = zero;
				for (int i = 0; i < localIdx; i += TMPPERLL4) {
					int cnt = min((int)atomicAdd(indexes + row * NX + col, TMPPERLL4), (int)(maxOut - TMPPERLL4));
					buffer[((ulong)(row * NX + col) * maxOut + cnt) / TMPPERLL4] = Pack4edges(tmp[row][i], tmp[row][i+1], tmp[row][i+2], tmp[row][i+3]);//*(ulong4 *)(&tmp[row][i]);
				}
			}
		}


 //   template<int maxOut, typename EdgeOut>
        __global__ void SeedB(const siphash_keys *sipkeys, const uint2 * __restrict__ source, ulong4 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes, const int halfA, const int halfE, int maxOut) {
            const int group = blockIdx.x;
            const int dim = blockDim.x;
            const int lid = threadIdx.x;
            //const int FLUSHB2 = 2 * FLUSHB;

            __shared__ uint2 tmp[NX][FLUSHB2];
            const int TMPPERLL4 = sizeof(ulong4) / sizeof(uint2);
            __shared__ int counters[NX];

            for (int col = lid; col < NX; col += dim)
                counters[col] = 0;
            __syncthreads();
            const int row = group / NX;
            const int bucketEdges = min((int)sourceIndexes[group + halfE], (int)maxOut);
            const int loops = (bucketEdges + dim-1) / dim;
            for (int loop = 0; loop < loops; loop++) {
                int col; int counter = 0;
                const int edgeIndex = loop * dim + lid;
                if (edgeIndex < bucketEdges) {
                    const int index = group * maxOut + edgeIndex;
                    //uint2 edge = __ldg(&source[index + halfA/sizeof(uint2)]);
		    uint2 edge = source[index + halfA/sizeof(uint2)];
                    if (null(edge)) continue;
                    u32 node1 = endpoint(sipkeys, edge, 0);
                    col = (node1 >> XBITS) & XMASK;
                    counter = min((int)atomicAdd(counters + col, 1), (int)(FLUSHB2-1));
                    tmp[col][counter] = edge;
                }
                __syncthreads();
                if (counter == FLUSHB-1) {
                    int localIdx = min(FLUSHB2, counters[col]);
                    int newCount = localIdx % FLUSHB;
                    int nflush = localIdx - newCount;
                    int cnt = min((int)atomicAdd(destinationIndexes + row * NX + col + halfE, nflush), (int)(maxOut - nflush));
                    for (int i = 0; i < nflush; i += TMPPERLL4)
                        destination[((ulong)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4 + halfA/sizeof(ulong4)] = Pack4edges(tmp[col][i], tmp[col][i+1], tmp[col][i+2], tmp[col][i+3]);//*(ulong4 *)(&tmp[col][i]);
                    for (int t = 0; t < newCount; t++) {
                        tmp[col][t] = tmp[col][t + nflush];
                    }
                    counters[col] = newCount;
                }
                __syncthreads();
            }
            uint2 zero = make_Edge_by_node(0, tmp[0][0], 0, 0);
            for (int col = lid; col < NX; col += dim) {
                int localIdx = min(FLUSHB2, counters[col]);
                for (int j = localIdx; j % TMPPERLL4; j++)
                    tmp[col][j] = zero;
                for (int i = 0; i < localIdx; i += TMPPERLL4) {
                    int cnt = min((int)atomicAdd(destinationIndexes + row * NX + col + halfE, TMPPERLL4), (int)(maxOut - TMPPERLL4));
                    destination[((ulong)(row * NX + col) * maxOut + cnt) / TMPPERLL4 + halfA/sizeof(ulong4)] = Pack4edges(tmp[col][i], tmp[col][i+1], tmp[col][i+2], tmp[col][i+3]);//*(ulong4 *)(&tmp[col][i]);
                }
            }
        }

//    template<int maxIn, typename EdgeIn, int maxOut, typename EdgeOut>
        __global__ void Round(const int round, const siphash_keys *sipkeys, const uint2 * __restrict__ source, uint2 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes, int maxIn, int maxOut) {
            const int group = blockIdx.x;
            const int dim = blockDim.x;
            const int lid = threadIdx.x;
            //const static int COUNTERWORDS = NZ / 16; // 16 2-bit counters per 32-bit word

            __shared__ u32 ecounters[COUNTERWORDS];

            for (int i = lid; i < COUNTERWORDS; i += dim)
                ecounters[i] = 0;
            __syncthreads();
            const int edgesInBucket = min(sourceIndexes[group], maxIn);
            const int loops = (edgesInBucket + dim-1) / dim;

            for (int loop = 0; loop < loops; loop++) {
                const int lindex = loop * dim + lid;
                if (lindex < edgesInBucket) {
                    const int index = maxIn * group + lindex;
//                    uint2 edge = __ldg(&source[index]);
		    uint2 edge = source[index];
                    if (null(edge)) continue;
                    u32 node = endpoint(sipkeys, edge, round&1);
                    Increase2bCounter(ecounters, node >> (2*XBITS));
                }
            }
            __syncthreads();
            for (int loop = 0; loop < loops; loop++) {
                const int lindex = loop * dim + lid;
                if (lindex < edgesInBucket) {
                    const int index = maxIn * group + lindex;
//                    uint2 edge = __ldg(&source[index]);
		    uint2 edge = source[index];
                    if (null(edge)) continue;
                    u32 node0 = endpoint(sipkeys, edge, round&1);
                    if (Read2bCounter(ecounters, node0 >> (2*XBITS))) {
                        u32 node1 = endpoint(sipkeys, edge, (round&1)^1);
                        const int bucket = node1 & X2MASK;
                        const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), maxOut - 1);
                        destination[bucket * maxOut + bktIdx] = edge;//(round&1) ? make_Edge(edge, *destination, node1, node0)
                            //: make_Edge(edge, *destination, node0, node1);
                    }
                }
            }
        }

	__global__ void Tail(const uint2 *source, uint2 *destination, const int *sourceIndexes, int *destinationIndexes, int maxIn) {
	  const int lid = threadIdx.x;
	  const int group = blockIdx.x;
	  const int dim = blockDim.x;
	  int myEdges = sourceIndexes[group];
	  __shared__ int destIdx;

	  if (lid == 0)
		destIdx = atomicAdd(destinationIndexes, myEdges);

	  __syncthreads();
	  for (int i = lid; i < myEdges; i += dim)
		destination[destIdx + lid] = source[group * maxIn + lid];
	}


    edgetrimmer::edgetrimmer(const trimparams _tp) {
        indexesSize = NX * NY * sizeof(u32);
        tp = _tp;

        checkCudaErrors(cudaMalloc((void**)&dipkeys, sizeof(siphash_keys)));
        checkCudaErrors(cudaMalloc((void**)&indexesE, indexesSize));
        checkCudaErrors(cudaMalloc((void**)&indexesE2, indexesSize));
	checkCudaErrors(cudaMalloc((void**)&recoveredges, sizeof(uint2)*PROOFSIZE));

        sizeA = ROW_EDGES_A * NX * sizeof(uint2);
        sizeB = ROW_EDGES_B * NX * sizeof(uint2);

        const size_t bufferSize = sizeA + sizeB;
        fprintf(stderr, "bufferSize: %lu\n", bufferSize);
        checkCudaErrors(cudaMalloc((void**)&bufferA, sizeA));
	checkCudaErrors(cudaMalloc((void**)&bufferB, sizeB));
	checkCudaErrors(cudaMalloc((void**)&bufferAB, sizeA));
    }
    ulong edgetrimmer::globalbytes() const {
        return (sizeA+sizeB) + 2 * indexesSize + sizeof(siphash_keys);
    }
    edgetrimmer::~edgetrimmer() {
        cudaFree(bufferA);
        cudaFree(indexesE2);
        cudaFree(indexesE);
        cudaFree(dipkeys);
        cudaDeviceReset();
    }


void saveFile(ulong *v, int n, char* filename){
	FILE *fp = fopen(filename, "w");
	if(fp == NULL){
		printf("open file error\n");
		return;
	}
	for(int i = 0; i < n; i++){
		fprintf(fp, "%ld\n", v[i]);
	}
	fclose(fp);
}
int compare(const void *a, const void *b){
	return (*(ulong*)a - *(ulong*)b);
}

    u32 edgetrimmer::trim(uint32_t device) {
        cudaSetDevice(device);

        cudaMemset(indexesE, 0, indexesSize);
        cudaMemset(indexesE2, 0, indexesSize);
        cudaMemcpy(dipkeys, &sipkeys, sizeof(sipkeys), cudaMemcpyHostToDevice);

        checkCudaErrors(cudaDeviceSynchronize());

	SeedA<<<tp.genA.blocks, tp.genA.tpb>>>(dipkeys, bufferAB, (int *)indexesE, EDGES_A);

        const u32 halfA = sizeA/2 ;/// sizeof(ulong4);
        const u32 halfE = NX2 / 2;
	SeedB<<<tp.genB.blocks/2, tp.genB.tpb>>>(dipkeys, (const uint2 *)bufferAB, bufferA, (const int *)indexesE, indexesE2, 0, 0, EDGES_A);
	SeedB<<<tp.genB.blocks/2, tp.genB.tpb>>>(dipkeys, (const uint2 *)(bufferAB), bufferA, (const int *)(indexesE), indexesE2, halfA, halfE, EDGES_A);

	cudaMemset(indexesE, 0, indexesSize);
	Round<<<tp.trim.blocks, tp.trim.tpb>>>(0, dipkeys, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE, EDGES_A, EDGES_B); // to .632

	cudaMemset(indexesE2, 0, indexesSize);
	Round<<<tp.trim.blocks, tp.trim.tpb>>>(1, dipkeys, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2, EDGES_B, EDGES_B/2); // to .296

	cudaMemset(indexesE, 0, indexesSize);
	Round<<<tp.trim.blocks, tp.trim.tpb>>>(2, dipkeys, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE, EDGES_B/2, EDGES_A/4); // to .176
	cudaMemset(indexesE2, 0, indexesSize);
	Round<<<tp.trim.blocks, tp.trim.tpb>>>(3, dipkeys, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2, EDGES_A/4, EDGES_B/4); // to .117 


        cudaDeviceSynchronize();

        for (int round = 4; round < tp.ntrims; round += 2) {
		cudaMemset(indexesE, 0, indexesSize);
		Round<<<tp.trim.blocks, tp.trim.tpb>>>(round, dipkeys,  (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE, EDGES_B/4, EDGES_B/4);
		cudaMemset(indexesE2, 0, indexesSize);
		Round<<<tp.trim.blocks, tp.trim.tpb>>>(round+1, dipkeys,  (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2, EDGES_B/4, EDGES_B/4);
        }

        checkCudaErrors(cudaDeviceSynchronize()); 

        cudaMemset(indexesE, 0, indexesSize);
        checkCudaErrors(cudaDeviceSynchronize()); 

        Tail<<<tp.tail.blocks, tp.tail.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE, DUCK_B_EDGES/4);
        cudaMemcpy(hostA, indexesE, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);

        checkCudaErrors(cudaDeviceSynchronize());
	fprintf(stderr, "Host A [0]: %zu\n", hostA[0]);
        return hostA[0];
    }


};
