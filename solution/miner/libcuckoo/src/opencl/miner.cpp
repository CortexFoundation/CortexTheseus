#include "cuckoo_solver.hpp"
#include "cuckaroo_solver.hpp"
//#include "kernel_source.h"
#include "gold_miner.h"
#include "../../miner.h"
#include <vector>
#include "monitor.hpp"
#include "../cuckoo.h"

//cuckoogpu::cuckoo_solver_ctx* ctx = NULL;
//cuckoogpu::cuckaroo_solver_ctx* cuckaroo_ctx = NULL;
std::vector<cuckoogpu::solver_ctx*> ctx;


void getDeviceInfo(){
	getAllDeviceInfo();
	//device count
	//driver version
	//devices memory,compute units, capability, major.minor
}
int32_t FindSolutionsByGPU(
        uint8_t *header,
        uint64_t nonce,
    uint32_t threadId)
{
    using namespace cuckoogpu;
    using std::vector;
    // printf("[CuckooFind, sols.size()SolutionsCuda] thread: %d\n", getpid());
    //cudaSetDevice(ctx[threadId]->device);
    ctx[threadId]->setheadernonce((char*)header, nonce);

    char headerInHex[65];
    for (uint32_t i = 0; i < 32; i++) {
        sprintf(headerInHex + 2 * i, "%02x", *((unsigned int8_t*)(header + i)));
    }
    headerInHex[64] = '\0';

    // printf("Looking for %d-cycle on cuckoo%d(\"%s\",%019lu)\n", PROOFSIZE, NODEBITS, headerInHex,  nonce);
    u32 nsols = ctx[threadId]->solve();
    /*vector<vector<u32> > sols;
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
*/
    return nsols;
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
	cl_platform_id platformId = getOnePlatform ();
	if (platformId == NULL)
		return;
//	getPlatformInfo (platformId);
	cl_device_id deviceId = getOneDevice (platformId, device);
	if (deviceId == NULL)
		exit(0);
	cl_context context = createContext (platformId, deviceId);
	if (context == NULL)
		return;
	cl_command_queue commandQueue = createCommandQueue (context, deviceId);
	if (commandQueue == NULL)
		return;

	string sourceStr = get_kernel_source();
	size_t size = sourceStr.size();
	const char *source = sourceStr.c_str ();
	cl_program program = createProgram (context, &source, size);

//	cl_program program = createByBinaryFile("trimmer.bin", context, deviceId);
	if (program == NULL){
		printf("create program error\n");
		return;
	}
//	printf("EDGEBITS = %d, PROOFSIZE = %d, EXPAND=%d\n", EDGEBITS, PROOFSIZE, tp.expand);
	char options[1024] = "-I./";
	sprintf (options, "-cl-std=CL2.0 -I./ -DEDGEBITS=%d -DPROOFSIZE=%d  -DEXPAND=%d", EDGEBITS, PROOFSIZE, tp.expand);

	buildProgram (program, &(deviceId), options);
//	saveBinaryFile(program, deviceId);
	cl_ulong maxThreadsPerBlock = 0;
	clGetDeviceInfo (deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (maxThreadsPerBlock), &maxThreadsPerBlock, NULL);
	assert (tp.genA.tpb <= maxThreadsPerBlock);
	assert (tp.genB.tpb <= maxThreadsPerBlock);
	assert (tp.trim.tpb <= maxThreadsPerBlock);
	assert (tp.tail.tpb <= maxThreadsPerBlock);
	assert (tp.recover.tpb <= maxThreadsPerBlock);
	ctx[index]->init(tp, device, context, commandQueue, program);
  //  printf("50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, tp.ntrims, NX);
  //  u64 bytes = ctx[index]->trimmer->globalbytes();

  /*  int unit;
    for (unit=0; bytes >= 10240; bytes>>=10,unit++);
    printf("Using %d%cB of global memory.\n", (u32)bytes, " KMGT"[unit]);
    */
}

void CuckooInitialize(uint32_t* devices, uint32_t deviceNum, int selected = 0, int printDeviceInfo = 1) {
    using namespace cuckoogpu;
    using std::vector;
    if(printDeviceInfo != 0)
	getDeviceInfo();
#ifdef PLATFORM_AMD
    if(monitor_init(deviceNum) < 0) exit(0);
#endif
    for(uint i = 0; i < deviceNum; i++){
            if(selected == 0){
                    ctx.push_back(new cuckoo_solver_ctx());
            }else{
                    ctx.push_back(new cuckaroo_solver_ctx());
            }
            initOne(i, devices[i]);
    }
}

int monitor(unsigned int device_count, unsigned int *fanSpeeds, unsigned int *temperatures){
#ifdef PLATFORM_AMD
	return query_fan_tem(device_count, fanSpeeds, temperatures);
#else
	return 0;
#endif
}
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
