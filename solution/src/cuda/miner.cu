#include "cuckoo_solver.hpp"
#include "cuckaroo_solver.hpp"
#include "../../miner.h"
#include <vector>
#include "monitor.hpp"
#include "../cuckoo.h"

std::vector<cuckoogpu::solver_ctx*> ctx;

void getDeviceInfo(){
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if(error_id != cudaSuccess){
		printf("get device count error : %s\n", cudaGetErrorString(error_id));
		return;
	}
	
	if(deviceCount == 0){
		printf("there are no available device that supprot CUDA\n");
	}

	printf("NVIDIA Cards available: %d\n", deviceCount);
	int driverVersion = 0, runtimeVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("\033[0;32;40m CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
	for(int dev = 0; dev < deviceCount; ++dev){
			cudaSetDevice(dev);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);
			size_t freeSize, totalSize;
			cudaMemGetInfo(&freeSize, &totalSize);
		printf("\033[0;32;40m GPU #%d: %s, total %.0fMB, free %.0fMB, %u compute units, capability: %d.%d\033[0m \n", dev, deviceProp.name, (float)deviceProp.totalGlobalMem/1048576.0f, (float)freeSize/1048576.0f, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
		cudaDeviceReset();
	}
}


int32_t FindSolutionsByGPU(
        uint8_t *header,
        uint64_t nonce,
    uint32_t threadId
    )
{
    using namespace cuckoogpu;
    using std::vector;
    // printf("[CuckooFind, sols.size()SolutionsCuda] thread: %d\n", getpid());
    cudaSetDevice(ctx[threadId]->device);
    ctx[threadId]->setheadernonce((char*)header, nonce);

    char headerInHex[65];
    for (uint32_t i = 0; i < 32; i++) {
        sprintf(headerInHex + 2 * i, "%02x", *((unsigned int8_t*)(header + i)));
    }
    headerInHex[64] = '\0';

    // printf("Looking for %d-cycle on cuckoo%d(\"%s\",%019lu)\n", PROOFSIZE, NODEBITS, headerInHex,  nonce);
    u32 nedges = ctx[threadId]->solve();
    return nedges;
}

int32_t FindCycles(
	uint32_t threadId, 
	uint32_t nedges,
        uint32_t *result,
        uint32_t resultBuffSize,
        uint32_t *solLength,
        uint32_t *numSol){

    using namespace cuckoogpu;
    using std::vector;
    cudaSetDevice(ctx[threadId]->device);
    u32 nsols = ctx[threadId]->findcycles(nedges);
    vector<vector<u32> > sols;
    vector<vector<u32> >* solutions = &sols;
    for (unsigned s = 0; s < nsols; s++) {
        u32* prf = &(ctx[threadId]->sols[s * PROOFSIZE]);
        solutions->push_back(vector<u32>());
        vector<u32>& sol = solutions->back();
        for (uint32_t idx = 0; idx < PROOFSIZE; idx++) {
            sol.push_back(prf[idx]);
        }
        // std::sort(sol.begin(), sol.end());
    }
    *solLength = 0;
    *numSol = sols.size();
    if (sols.size() == 0)
        return 0;
    *solLength = uint32_t(sols[0].size());
    for (size_t n = 0; n < min(sols.size(), (size_t)resultBuffSize / (*solLength)); n++)
    {
        vector<u32>& sol = sols[n];
        for (size_t i = 0; i < sol.size(); i++) {
            result[i + n * (*solLength)] = sol[i];
        }
    }

    return nsols > 0;
}

void initOne(uint32_t index, uint32_t device){
    using namespace cuckoogpu;
    using std::vector;
    trimparams tp;
    int nDevices = 0;
    checkCudaErrors(cudaGetDeviceCount(&nDevices));
    assert(device < nDevices);
    cudaSetDevice(device);
    // printf("Cuckoo: Device ID %d\n", device);
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));
    assert(tp.genA.tpb <= prop.maxThreadsPerBlock);
    assert(tp.genB.tpb <= prop.maxThreadsPerBlock);
    assert(tp.trim.tpb <= prop.maxThreadsPerBlock);
    // assert(tp.tailblocks <= prop.threadDims[0]);
    assert(tp.tail.tpb <= prop.maxThreadsPerBlock);
    assert(tp.recover.tpb <= prop.maxThreadsPerBlock);
    //ctx = new solver_ctx(tp, device);
    ctx[index]->init(tp, device);

   // printf("50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, tp.ntrims, NX);
    u64 bytes = ctx[index]->trimmer->globalbytes();

    int unit;
    for (unit=0; bytes >= 10240; bytes>>=10,unit++);
    //printf("Using %d%cB of global memory.\n", (u32)bytes, " KMGT"[unit]);
}

void CuckooInitialize(uint32_t* devices, uint32_t deviceNum, int selected = 0, int printDeviceInfo = 1) {
    //printf("thread: %d\n", getpid());
    using namespace cuckoogpu;
    using std::vector;
    if(printDeviceInfo != 0)
   	getDeviceInfo();
		int ret = monitor_init(deviceNum);
    if(ret < 0) exit(0);

    for(int i = 0; i < deviceNum; i++){
				if(devices[i] >= ret){
					printf("the device id %d must less than max device number %d\n", devices[i], ret);
					exit(0);
				}
				ctx.push_back(new cuckaroo_solver_ctx());
				initOne(i, devices[i]);
    }
}

int monitor(unsigned int device_count, unsigned int *fanSpeeds, unsigned int *temperatures){
	return query_fan_tem(device_count, fanSpeeds, temperatures);	
}
//void CuckooFinalizeCPU(){
//}
void CuckooFinalize(){
}

int32_t CuckooVerifyProof(uint8_t *header, uint64_t nonce, result_t *result) {
    using namespace cuckoogpu;
    uint64_t littleEndianNonce = htole64(nonce);
    char headerBuf[40];
    memcpy(headerBuf, header, 32);
    memcpy(headerBuf + 32, (uint64_t*)(&littleEndianNonce), sizeof(nonce));
    siphash_keys key;
    setheader(headerBuf, 40, &key);
    int res = verify_proof(result, &key);
    return res;
}

int32_t CuckooVerifyProof_cuckaroo(uint8_t *header, uint64_t nonce, result_t *result) {
    using namespace cuckoogpu;
    uint64_t littleEndianNonce = htole64(nonce);
    char headerBuf[40];
    memcpy(headerBuf, header, 32);
    memcpy(headerBuf + 32, (uint64_t*)(&littleEndianNonce), sizeof(nonce));
    siphash_keys key;
    setheader(headerBuf, 40, &key);
    int res = verify_proof_cuckaroo(result, &key);
    return res;
}
