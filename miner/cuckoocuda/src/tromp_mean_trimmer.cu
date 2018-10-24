#include "trimmer.h" 

namespace cuckoogpu {
#include "siphash.cuh"

__constant__ uint2 recoveredges[PROOFSIZE];

__global__ void Recovery(const siphash_keys &sipkeys, ulonglong4 *buffer, int *indexes) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  const int loops = NEDGES / nthreads;
  __shared__ u32 nonces[PROOFSIZE];

  if (lid < PROOFSIZE) nonces[lid] = 0;
  __syncthreads();
  for (int i = 0; i < loops; i++) {
	u64 nonce = gid * loops + i;
	u64 u = dipnode(sipkeys, nonce, 0);
	u64 v = dipnode(sipkeys, nonce, 1);
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

    __constant__ uint2 e0 = {0,0};

    __device__ __forceinline__ ulonglong4 Pack8(const u32 e0, const u32 e1, const u32 e2, const u32 e3, const u32 e4, const u32 e5, const u32 e6, const u32 e7) {
        return make_ulonglong4((u64)e0<<32|e1, (u64)e2<<32|e3, (u64)e4<<32|e5, (u64)e6<<32|e7);
    }

    __device__ bool null(u32 nonce) {
        return nonce == 0;
    }

    __device__ bool null(uint2 nodes) {
        return nodes.x == 0 && nodes.y == 0;
    }
    template<int maxOut, typename EdgeOut>
        __global__ void SeedA(const siphash_keys &sipkeys, ulonglong4 * __restrict__ buffer, int * __restrict__ indexes) {
            const int group = blockIdx.x;
            const int dim = blockDim.x;
            const int lid = threadIdx.x;
            const int gid = group * dim + lid;
            const int nthreads = gridDim.x * dim;
            const int FLUSHA2 = 2*FLUSHA;

            __shared__ EdgeOut tmp[NX][FLUSHA2]; // needs to be ulonglong4 aligned
            const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(EdgeOut);
            __shared__ int counters[NX];

            for (int row = lid; row < NX; row += dim)
                counters[row] = 0;
            __syncthreads();

            const int col = group % NX;
            const int loops = NEDGES / nthreads;
            for (int i = 0; i < loops; i++) {
                u32 nonce = gid * loops + i;
                u32 node1, node0 = dipnode(sipkeys, (u64)nonce, 0);
                if (sizeof(EdgeOut) == sizeof(uint2))
                    node1 = dipnode(sipkeys, (u64)nonce, 1);
                int row = node0 & XMASK;
                int counter = min((int)atomicAdd(counters + row, 1), (int)(FLUSHA2-1));
                tmp[row][counter] = make_Edge(nonce, tmp[0][0], node0, node1);
                __syncthreads();
                if (counter == FLUSHA-1) {
                    int localIdx = min(FLUSHA2, counters[row]);
                    int newCount = localIdx % FLUSHA;
                    int nflush = localIdx - newCount;
                    int cnt = min((int)atomicAdd(indexes + row * NX + col, nflush), (int)(maxOut - nflush));
                    for (int i = 0; i < nflush; i += TMPPERLL4)
                        buffer[((u64)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
                    for (int t = 0; t < newCount; t++) {
                        tmp[row][t] = tmp[row][t + nflush];
                    }
                    counters[row] = newCount;
                }
                __syncthreads();
            }
            EdgeOut zero = make_Edge(0, tmp[0][0], 0, 0);
            for (int row = lid; row < NX; row += dim) {
                int localIdx = min(FLUSHA2, counters[row]);
                for (int j = localIdx; j % TMPPERLL4; j++)
                    tmp[row][j] = zero;
                for (int i = 0; i < localIdx; i += TMPPERLL4) {
                    int cnt = min((int)atomicAdd(indexes + row * NX + col, TMPPERLL4), (int)(maxOut - TMPPERLL4));
                    buffer[((u64)(row * NX + col) * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
                }
            }
        }
    template<int maxIn, typename EdgeIn, int maxOut, typename EdgeOut>
        __global__ void Round(const int round, const siphash_keys &sipkeys, const EdgeIn * __restrict__ source, EdgeOut * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
            const int group = blockIdx.x;
            const int dim = blockDim.x;
            const int lid = threadIdx.x;
            const static int COUNTERWORDS = NZ / 16; // 16 2-bit counters per 32-bit word

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
                    EdgeIn edge = __ldg(&source[index]);
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
                    EdgeIn edge = __ldg(&source[index]);
                    if (null(edge)) continue;
                    u32 node0 = endpoint(sipkeys, edge, round&1);
                    if (Read2bCounter(ecounters, node0 >> (2*XBITS))) {
                        u32 node1 = endpoint(sipkeys, edge, (round&1)^1);
                        const int bucket = node1 & X2MASK;
                        const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), maxOut - 1);
                        destination[bucket * maxOut + bktIdx] = (round&1) ? make_Edge(edge, *destination, node1, node0)
                            : make_Edge(edge, *destination, node0, node1);
                    }
                }
            }
            // if (group==0&&lid==0) printf("round %d cnt(0,0) %d\n", round, sourceIndexes[0]);
        }
    template<int maxOut, typename EdgeOut>
        __global__ void SeedB(const siphash_keys &sipkeys, const EdgeOut * __restrict__ source, ulonglong4 * __restrict__ destination, const int * __restrict__ sourceIndexes, int * __restrict__ destinationIndexes) {
            const int group = blockIdx.x;
            const int dim = blockDim.x;
            const int lid = threadIdx.x;
            const int FLUSHB2 = 2 * FLUSHB;

            __shared__ EdgeOut tmp[NX][FLUSHB2];
            const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(EdgeOut);
            __shared__ int counters[NX];

            // if (group>=0&&lid==0) printf("group  %d  -\n", group);
            for (int col = lid; col < NX; col += dim)
                counters[col] = 0;
            __syncthreads();
            const int row = group / NX;
            const int bucketEdges = min((int)sourceIndexes[group], (int)maxOut);
            const int loops = (bucketEdges + dim-1) / dim;
            for (int loop = 0; loop < loops; loop++) {
                int col; int counter = 0;
                const int edgeIndex = loop * dim + lid;
                if (edgeIndex < bucketEdges) {
                    const int index = group * maxOut + edgeIndex;
                    EdgeOut edge = __ldg(&source[index]);
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
                    int cnt = min((int)atomicAdd(destinationIndexes + row * NX + col, nflush), (int)(maxOut - nflush));
                    for (int i = 0; i < nflush; i += TMPPERLL4)
                        destination[((u64)(row * NX + col) * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
                    for (int t = 0; t < newCount; t++) {
                        tmp[col][t] = tmp[col][t + nflush];
                    }
                    counters[col] = newCount;
                }
                __syncthreads();
            }
            EdgeOut zero = make_Edge(0, tmp[0][0], 0, 0);
            for (int col = lid; col < NX; col += dim) {
                int localIdx = min(FLUSHB2, counters[col]);
                for (int j = localIdx; j % TMPPERLL4; j++)
                    tmp[col][j] = zero;
                for (int i = 0; i < localIdx; i += TMPPERLL4) {
                    int cnt = min((int)atomicAdd(destinationIndexes + row * NX + col, TMPPERLL4), (int)(maxOut - TMPPERLL4));
                    destination[((u64)(row * NX + col) * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
                }
            }
        }

	template<int maxIn>
		__global__ void Tail(const uint2 *source, uint2 *destination, const int *sourceIndexes, int *destinationIndexes) {
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

    __device__ u32 endpoint(const siphash_keys &sipkeys, u32 nonce, int uorv) {
        return dipnode(sipkeys, nonce, uorv);
    }

    __device__ u32 endpoint(const siphash_keys &sipkeys, uint2 nodes, int uorv) {
        return uorv ? nodes.y : nodes.x;
    }

    __device__ uint2 make_Edge(const u32 nonce, const uint2 dummy, const u32 node0, const u32 node1) {
        return make_uint2(node0, node1);
    }

    __device__ uint2 make_Edge(const uint2 edge, const uint2 dummy, const u32 node0, const u32 node1) {
        return edge;
    }

    __device__ u32 make_Edge(const u32 nonce, const u32 dummy, const u32 node0, const u32 node1) {
        return nonce;
    }

    edgetrimmer::edgetrimmer(const trimparams _tp) {
        indexesSize = NX * NY * sizeof(u32);
        tp = _tp;
        checkCudaErrors(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
        checkCudaErrors(cudaMalloc((void**)&uvnodes, PROOFSIZE * 2 * sizeof(u32)));
        checkCudaErrors(cudaMalloc((void**)&dipkeys, sizeof(siphash_keys)));
        checkCudaErrors(cudaMalloc((void**)&indexesE, indexesSize));
        checkCudaErrors(cudaMalloc((void**)&indexesE2, indexesSize));
        sizeA = ROW_EDGES_A * NX * (tp.expand > 0 ? sizeof(u32) : sizeof(uint2));
        sizeB = ROW_EDGES_B * NX * (tp.expand > 1 ? sizeof(u32) : sizeof(uint2));
        const size_t bufferSize = sizeA + sizeB;
        printf("bufferSize: %lu\n", bufferSize);
        checkCudaErrors(cudaMalloc((void**)&bufferA, bufferSize));
        bufferB  = bufferA + sizeA / sizeof(ulonglong4);
        bufferAB = bufferA + sizeB / sizeof(ulonglong4);
    }
    u64 edgetrimmer::globalbytes() const {
        return (sizeA+sizeB) + 2 * indexesSize + sizeof(siphash_keys) + PROOFSIZE * 2 * sizeof(u32) + sizeof(edgetrimmer);
    }
    edgetrimmer::~edgetrimmer() {
        cudaFree(bufferA);
        cudaFree(indexesE2);
        cudaFree(indexesE);
        cudaFree(dipkeys);
        cudaFree(uvnodes);
        cudaFree(dt);
        cudaDeviceReset();
    }
    u32 edgetrimmer::trim(uint32_t device) {
        cudaSetDevice(device);
        cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
        cudaEvent_t start, stop;
        // cudaEvent_t startall, stopall;
        // checkCudaErrors(cudaEventCreate(&startall)); checkCudaErrors(cudaEventCreate(&stopall));
        checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));

        cudaMemset(indexesE, 0, indexesSize);
        cudaMemset(indexesE2, 0, indexesSize);
        cudaMemcpy(dipkeys, &sipkeys, sizeof(sipkeys), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        float durationA, durationB;
        cudaEventRecord(start, NULL);

		fprintf(stderr, "Tp block %zu and tpb %zu", tp.genA.blocks, tp.genA.tpb);
        if (tp.expand == 0)
            SeedA<EDGES_A, uint2><<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, bufferAB, (int *)indexesE);
        else
            SeedA<EDGES_A,   u32><<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, bufferAB, (int *)indexesE);

        checkCudaErrors(cudaDeviceSynchronize());
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop); cudaEventElapsedTime(&durationA, start, stop); cudaEventRecord(start, NULL);

        const u32 halfA = sizeA/2 / sizeof(ulonglong4);
        const u32 halfE = NX2 / 2;
        if (tp.expand == 0) {
            SeedB<EDGES_A, uint2><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const uint2 *)bufferAB, bufferA, (const int *)indexesE, indexesE2);
            SeedB<EDGES_A, uint2><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const uint2 *)(bufferAB+halfA), bufferA+halfA, (const int *)(indexesE+halfE), indexesE2+halfE);
        } else {
            SeedB<EDGES_A,   u32><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const   u32 *)bufferAB, bufferA, (const int *)indexesE, indexesE2);
            SeedB<EDGES_A,   u32><<<tp.genB.blocks/2, tp.genB.tpb>>>(*dipkeys, (const   u32 *)(bufferAB+halfA), bufferA+halfA, (const int *)(indexesE+halfE), indexesE2+halfE);
        }

        checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
        // printf("Seeding completed in %.0f + %.0f ms\n", durationA, durationB);

        cudaMemset(indexesE, 0, indexesSize);

        if (tp.expand == 0)
            Round<EDGES_A, uint2, EDGES_B, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632
        else if (tp.expand == 1)
            Round<EDGES_A,   u32, EDGES_B, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const   u32 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632
        else // tp.expand == 2
            Round<EDGES_A,   u32, EDGES_B,   u32><<<tp.trim.blocks, tp.trim.tpb>>>(0, *dipkeys, (const   u32 *)bufferA, (  u32 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .632

        cudaMemset(indexesE2, 0, indexesSize);

        if (tp.expand < 2)
            Round<EDGES_B, uint2, EDGES_B/2, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(1, *dipkeys, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .296
        else
            Round<EDGES_B,   u32, EDGES_B/2, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(1, *dipkeys, (const   u32 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .296

        cudaMemset(indexesE, 0, indexesSize);
        Round<EDGES_B/2, uint2, EDGES_A/4, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(2, *dipkeys, (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE); // to .176
        cudaMemset(indexesE2, 0, indexesSize);
        Round<EDGES_A/4, uint2, EDGES_B/4, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(3, *dipkeys, (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2); // to .117

        cudaDeviceSynchronize();

        for (int round = 4; round < tp.ntrims; round += 2) {
            cudaMemset(indexesE, 0, indexesSize);
            Round<EDGES_B/4, uint2, EDGES_B/4, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(round, *dipkeys,  (const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE);
            cudaMemset(indexesE2, 0, indexesSize);
            Round<EDGES_B/4, uint2, EDGES_B/4, uint2><<<tp.trim.blocks, tp.trim.tpb>>>(round+1, *dipkeys,  (const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE, (int *)indexesE2);
        }

        cudaMemset(indexesE, 0, indexesSize);
        cudaDeviceSynchronize();

        Tail<EDGES_B/4><<<tp.tail.blocks, tp.tail.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE2, (int *)indexesE);
        cudaMemcpy(hostA, indexesE, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
		fprintf(stderr, "Host A [0]: %zu\n", hostA[0]);
        return hostA[0];
    }


};
