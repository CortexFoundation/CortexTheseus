#include "cuda_profiler_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <xmmintrin.h>
#include <atomic>
#include <thread>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trimmer.h"

namespace cuckoogpu {
	
__constant__ uint2 recoveredges[PROOFSIZE];

__global__ void Recovery(const siphash_keys &sipkeys, ulonglong4 *buffer, int *indexes) {
  const int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const int lid = threadIdx.x;
  /* const int nthreads = blockDim.x * gridDim.x; */
  __shared__ u32 nonces[PROOFSIZE];

  if (lid < PROOFSIZE) nonces[lid] = 0;
  __syncthreads();
  for (int i = 0; i < (1024 * 4); i++) {
	u64 nonce = gid * (1024 * 4) + i;
    u64 u = dipnode(sipkeys, nonce, 0);
    u64 v = dipnode(sipkeys, nonce, 1);
    for (int i = 0; i < PROOFSIZE; i++) {
      if ((recoveredges[i].x == u && recoveredges[i].y == v) ||
		// TODO compatiable
		  (recoveredges[i].y == u && recoveredges[i].x == v))
        nonces[i] = nonce;
    }
  }
  __syncthreads();
  if (lid < PROOFSIZE) {
    if (nonces[lid] > 0)
      indexes[lid] = nonces[lid];
  }
}

__device__ node_t dipnode(const siphash_keys &keys, edge_t nce, u32 uorv) {
  u64 nonce = 2*nce + uorv;
  u64 v0 = keys.k0, v1 = keys.k1, v2 = keys.k2, v3 = keys.k3^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1 ^ v2  ^ v3) & EDGEMASK;
}

/* #define DUCK_SIZE_A 130LL */
/* #define DUCK_SIZE_B 85LL */

/* #define DUCK_A_EDGES (DUCK_SIZE_A * 1024LL) */
/* #define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64LL) */
/* #define DUCK_B_EDGES (DUCK_SIZE_B * 1024LL) */
/* #define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64LL) */

#define DUCK_SIZE_A (EDGES_A / 1024LL)
#define DUCK_SIZE_B (EDGES_B / 1024LL)
#define DUCK_A_EDGES (DUCK_SIZE_A * 1024LL) 
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64LL)
#define DUCK_B_EDGES (DUCK_SIZE_B * 1024LL)
#define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64LL)

#ifndef EDGEBITS
#define EDGEBITS 29
#endif

#define CTHREADS 1024
#define BKTMASK4K (4096-1)

__device__ ulonglong4 Pack4edges(const uint2 e1, const  uint2 e2, const  uint2 e3, const  uint2 e4)
{
	u64 r1 = (((u64)e1.y << 32) | ((u64)e1.x));
	u64 r2 = (((u64)e2.y << 32) | ((u64)e2.x));
	u64 r3 = (((u64)e3.y << 32) | ((u64)e3.x));
	u64 r4 = (((u64)e4.y << 32) | ((u64)e4.x));
	return make_ulonglong4(r1, r2, r3, r4);
}

__global__  void FluffySeed2A(const siphash_keys &sipkeys, ulonglong4 * buffer, int * indexes)
{
	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;

	__shared__ uint2 tmp[64][15];
	__shared__ int counters[64];

	counters[lid] = 0;

	__syncthreads();

	for (int i = 0; i < 1024 * 16; i++)
	{
		u64 nonce = gid * (1024 * 16) + i;

		uint2 hash;

		/* hash.x = dipnode(v0i, v1i, v2i, v3i, nonce, 0); */
		hash.x = dipnode(sipkeys, nonce, 0);

		int bucket = hash.x & (64 - 1);

		__syncthreads();

		int counter = min((int)atomicAdd(counters + bucket, 1), (int)14);

		/* hash.y = dipnode(v0i, v1i, v2i, v3i, nonce, 1); */
		hash.y = dipnode(sipkeys, nonce, 1);

		tmp[bucket][counter] = hash;

		__syncthreads();

		{
			int localIdx = min(15, counters[lid]);

			if (localIdx >= 8)
			{
				int newCount = (localIdx - 8);
				counters[lid] = newCount;

				{
					int cnt = min((int)atomicAdd(indexes + lid, 8), (int)(DUCK_A_EDGES_64 - 8));

					{
						buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
						buffer[(lid * DUCK_A_EDGES_64 + cnt + 4) / 4] = Pack4edges(tmp[lid][4], tmp[lid][5], tmp[lid][6], tmp[lid][7]);
					}
				}

				for (int t = 0; t < newCount; t++)
				{
					tmp[lid][t] = tmp[lid][t + 8];
				}

			}
		}
	}

	__syncthreads();

	{
		int localIdx = min(16, counters[lid]);

		if (localIdx > 0)
		{
			int cnt = min((int)atomicAdd(indexes + lid, 4), (int)(DUCK_A_EDGES_64 - 4));
			buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(
				tmp[lid][0],
				localIdx > 1 ? tmp[lid][1] : make_uint2(0, 0),
				localIdx > 2 ? tmp[lid][2] : make_uint2(0, 0),
				localIdx > 3 ? tmp[lid][3] : make_uint2(0, 0));
		}
		if (localIdx > 4)
		{
			int cnt = min((int)atomicAdd(indexes + lid, 4), (int)(DUCK_A_EDGES_64 - 4));
			buffer[(lid * DUCK_A_EDGES_64 + cnt) / 4] = Pack4edges(
				tmp[lid][4],
				localIdx > 5 ? tmp[lid][5] : make_uint2(0, 0),
				localIdx > 6 ? tmp[lid][6] : make_uint2(0, 0),
				localIdx > 7 ? tmp[lid][7] : make_uint2(0, 0));
		}
	}

}

#define BKTGRAN 32
__global__  void FluffySeed2B(const  uint2 * source, ulonglong4 * destination, const  int * sourceIndexes, int * destinationIndexes, int startBlock)
{
	//const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;
	const int group = blockIdx.x;

	__shared__ uint2 tmp[64][15];
	__shared__ int counters[64];

	counters[lid] = 0;

	__syncthreads();

	const int offsetMem = startBlock * DUCK_A_EDGES_64;
	const int myBucket = group / BKTGRAN;
	const int microBlockNo = group % BKTGRAN;
	const int bucketEdges = min(sourceIndexes[myBucket + startBlock], (int)(DUCK_A_EDGES_64));
	const int microBlockEdgesCount = (DUCK_A_EDGES_64 / BKTGRAN);
	const int loops = (microBlockEdgesCount / 64);

	for (int i = 0; i < loops; i++)
	{
		int edgeIndex = (microBlockNo * microBlockEdgesCount) + (64 * i) + lid;

		if (edgeIndex < bucketEdges)
		{
			uint2 edge = source[offsetMem + (myBucket * DUCK_A_EDGES_64) + edgeIndex];

			if (edge.x == 0 && edge.y == 0) continue;

			int bucket = (edge.x >> 6) & (64 - 1);

			__syncthreads();

			int counter = min((int)atomicAdd(counters + bucket, 1), (int)14);

			tmp[bucket][counter] = edge;

			__syncthreads();

			int localIdx = min(15, counters[lid]);

			if (localIdx >= 8)
			{
				int newCount = (localIdx - 8);
				counters[lid] = newCount;

				{
					int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 8), (int)(DUCK_A_EDGES - 8));

					{
						destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(tmp[lid][0], tmp[lid][1], tmp[lid][2], tmp[lid][3]);
						destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt + 4) / 4] = Pack4edges(tmp[lid][4], tmp[lid][5], tmp[lid][6], tmp[lid][7]);
					}
				}

				for (int t = 0; t < newCount; t++)
				{
					tmp[lid][t] = tmp[lid][t + 8];
				}

			}
		}
	}

	__syncthreads();

	{
		int localIdx = min(16, counters[lid]);

		if (localIdx > 0)
		{
			int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 4), (int)(DUCK_A_EDGES - 4));
			destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(
				tmp[lid][0],
				localIdx > 1 ? tmp[lid][1] : make_uint2(0, 0),
				localIdx > 2 ? tmp[lid][2] : make_uint2(0, 0),
				localIdx > 3 ? tmp[lid][3] : make_uint2(0, 0));
		}
		if (localIdx > 4)
		{
			int cnt = min((int)atomicAdd(destinationIndexes + startBlock * 64 + myBucket * 64 + lid, 4), (int)(DUCK_A_EDGES - 4));
			destination[((myBucket * 64 + lid) * DUCK_A_EDGES + cnt) / 4] = Pack4edges(
				tmp[lid][4],
				localIdx > 5 ? tmp[lid][5] : make_uint2(0, 0),
				localIdx > 6 ? tmp[lid][6] : make_uint2(0, 0),
				localIdx > 7 ? tmp[lid][7] : make_uint2(0, 0));
		}
	}
}

__device__ __forceinline__  void Increase2bCounter(u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	u32 old = atomicOr(ecounters + word, mask) & mask;

	if (old > 0)
		atomicOr(ecounters + word + 4096, mask);
}

__device__ __forceinline__  bool Read2bCounter(u32 * ecounters, const int bucket)
{
	int word = bucket >> 5;
	unsigned char bit = bucket & 0x1F;
	u32 mask = 1 << bit;

	return (ecounters[word + 4096] & mask) > 0;
}

template<int bktInSize, int bktOutSize>
__global__   void FluffyRound(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes)
{
	//const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int lid = threadIdx.x;
	const int group = blockIdx.x;

	__shared__ u32 ecounters[8192];

	const int edgesInBucket = min(sourceIndexes[group], bktInSize);
	const int loops = (edgesInBucket + CTHREADS) / CTHREADS;

	ecounters[lid] = 0;
	ecounters[lid + 1024] = 0;
	ecounters[lid + (1024 * 2)] = 0;
	ecounters[lid + (1024 * 3)] = 0;
	ecounters[lid + (1024 * 4)] = 0;
	ecounters[lid + (1024 * 5)] = 0;
	ecounters[lid + (1024 * 6)] = 0;
	ecounters[lid + (1024 * 7)] = 0;

	__syncthreads();

	for (int i = 0; i < loops; i++) {
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket) {
			const int index = (bktInSize * group) + lindex;

			uint2 edge = source[index];
			if (edge.x == 0 && edge.y == 0) continue;

			Increase2bCounter(ecounters, (edge.x & EDGEMASK) >> 12);
		}
	}

	__syncthreads();

	for (int i = 0; i < loops; i++) {
		const int lindex = (i * CTHREADS) + lid;

		if (lindex < edgesInBucket) {
			const int index = (bktInSize * group) + lindex;

			uint2 edge = source[index];
			if (edge.x == 0 && edge.y == 0) continue;

			if (Read2bCounter(ecounters, (edge.x & EDGEMASK) >> 12)) {
				const int bucket = edge.y & BKTMASK4K;
				const int bktIdx = min(atomicAdd(destinationIndexes + bucket, 1), bktOutSize - 1);
				destination[(bucket * bktOutSize) + bktIdx] = make_uint2(edge.y, edge.x);
			}
		}
	}

}

template __global__ void FluffyRound<DUCK_A_EDGES, DUCK_B_EDGES>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES / 2>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 2>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 4>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);
template __global__ void FluffyRound<DUCK_B_EDGES / 4, DUCK_B_EDGES / 4>(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes);



__global__   void /*Magical*/FluffyTail/*Pony*/(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes)
{
	const int lid = threadIdx.x;
	const int group = blockIdx.x;

	int myEdges = sourceIndexes[group];
	__shared__ int destIdx;

	if (lid == 0)
		destIdx = atomicAdd(destinationIndexes, myEdges);

	__syncthreads();

	if (lid < myEdges)
	{
		destination[destIdx + lid] = source[group * DUCK_B_EDGES / 4 + lid];
	}
}

	edgetrimmer::edgetrimmer(const trimparams _tp) {
		tp = _tp;

		/* const size_t indexesSize = 128 * 128 * 4; */
        /* indexesSize = NX * NY * sizeof(u32); */
        indexesSize = 128 * 128 * sizeof(u32);
		/* const size_t bufferSize = DUCK_SIZE_A * 1024 * 4096 * 8; */
		/* const size_t bufferSize2 = DUCK_SIZE_A * 1024 * 4096 * 8; */
		sizeA = DUCK_SIZE_A * 1024 * 4096 * 8;
		sizeB = DUCK_SIZE_A * 1024 * 4096 * 8;

		checkCudaErrors(cudaMalloc((void**)&bufferA, sizeA));
		checkCudaErrors(cudaMalloc((void**)&bufferB, sizeB));
		
        checkCudaErrors(cudaMalloc((void**)&dipkeys, sizeof(siphash_keys)));

		checkCudaErrors(cudaMalloc((void**)&indexesE, indexesSize));
		checkCudaErrors(cudaMalloc((void**)&indexesE2, indexesSize));

		size_t free_device_mem = 0;
		size_t total_device_mem = 0;
		cudaMemGetInfo(&free_device_mem, &total_device_mem);
		fprintf(stderr, "Currently available amount of device memory: %zu bytes, request memory: %zu and %zu bytes\n", free_device_mem, sizeA, sizeB);
		fprintf(stderr, "Total amount of device memory: %zu bytes\n", total_device_mem);
		
		cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
		

		fprintf(stderr, "CUDA device armed\n\n\n");

	}

	u64 edgetrimmer::globalbytes() const {
        return (sizeA+sizeB) + 2 * indexesSize + sizeof(siphash_keys) + PROOFSIZE * 2 * sizeof(u32) + sizeof(edgetrimmer);
	}

	edgetrimmer::~edgetrimmer() {
		fprintf(stderr, "CUDA terminating...\n");
		fprintf(stderr, "#x\n");
		cudaFree(bufferA);
		cudaFree(bufferB);
        cudaFree(dipkeys);
		cudaFree(indexesE);
		cudaFree(indexesE2);
		cudaDeviceReset();
	}

	u32 edgetrimmer::trim(uint32_t device) {
		cudaError_t cudaStatus;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(device);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
			goto Error;
		}

		// loop starts here
		// wait for header hashes, nonce+r

		{
			cudaMemset(indexesE, 0, indexesSize);
			cudaMemset(indexesE2, 0, indexesSize);
			cudaMemcpy(dipkeys, &sipkeys, sizeof(sipkeys), cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();


			FluffySeed2A << < 512, 64 >> > (*dipkeys, (ulonglong4 *)bufferA, (int *)indexesE2);

			FluffySeed2B << < 32 * BKTGRAN, 64 >> > ((const uint2 *)bufferA, (ulonglong4 *)bufferB, (const int *)indexesE2, (int *)indexesE, 0);
			cudaMemcpy(bufferA, bufferB, sizeA / 2, cudaMemcpyDeviceToDevice);

			FluffySeed2B << < 32 * BKTGRAN, 64 >> > ((const uint2 *)bufferA, (ulonglong4 *)bufferB, (const int *)indexesE2, (int *)indexesE, 32);
			cudaStatus = cudaMemcpy(&((char *)bufferA)[sizeA / 2], bufferB, sizeA / 2, cudaMemcpyDeviceToDevice);

#ifdef DEBUG_INDEX_NUMBER
			cudaMemcpy(hostA, indexesE, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
			fprintf(stderr, "Index Number: %zu\n", hostA[0]);
#endif

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "status memcpy: %s\n", cudaGetErrorString(cudaStatus));

			cudaMemset(indexesE2, 0, indexesSize);
			FluffyRound<DUCK_A_EDGES, DUCK_B_EDGES> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
			cudaMemset(indexesE, 0, indexesSize);
			FluffyRound<DUCK_B_EDGES, DUCK_B_EDGES / 2> << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
			cudaMemset(indexesE2, 0, indexesSize);
			FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 2> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);
			cudaMemset(indexesE, 0, indexesSize);
			FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 2> << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
			cudaMemset(indexesE2, 0, indexesSize);
			FluffyRound<DUCK_B_EDGES / 2, DUCK_B_EDGES / 4> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);

#ifdef DEBUG_INDEX_NUMBER
			cudaMemcpy(hostA, indexesE2, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
			fprintf(stderr, "Index Number: %zu\n", hostA[0]);
#endif

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "status FluffyRound: %s\n", cudaGetErrorString(cudaStatus));


			cudaDeviceSynchronize();

			for (int i = 0; i < tp.ntrims; i += 2)
			{
				cudaMemset(indexesE, 0, indexesSize);
				FluffyRound<DUCK_B_EDGES / 4, DUCK_B_EDGES / 4> << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);
				cudaMemset(indexesE2, 0, indexesSize);
				FluffyRound<DUCK_B_EDGES / 4, DUCK_B_EDGES / 4> << < 4096, 1024 >> > ((const uint2 *)bufferA, (uint2 *)bufferB, (const int *)indexesE, (int *)indexesE2);

#ifdef DEBUG_INDEX_NUMBER
				cudaMemcpy(hostA, indexesE2, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
				fprintf(stderr, "Index Number: %zu\n", hostA[0]);
#endif

			}

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "status 80 FluffyRound: %s\n", cudaGetErrorString(cudaStatus));

			cudaMemset(indexesE, 0, indexesSize);
			cudaDeviceSynchronize();

			FluffyTail << < 4096, 1024 >> > ((const uint2 *)bufferB, (uint2 *)bufferA, (const int *)indexesE2, (int *)indexesE);

			cudaMemcpy(hostA, indexesE, NX * NY * sizeof(u32), cudaMemcpyDeviceToHost);
#ifdef DEBUG_INDEX_NUMBER
			fprintf(stderr, "Host A [0]: %zu\n", hostA[0]);
#endif


			cudaMemcpy(bufferB, bufferA, sizeA, cudaMemcpyDeviceToDevice);

			cudaDeviceSynchronize();

		}

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "status: %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}


		Error:
		fprintf(stderr, "Host A [0]: %zu\n", hostA[0]);
		return hostA[0];
	}

};
