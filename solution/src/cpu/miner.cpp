#include "cuckoo_mean.hpp"
#include "cuckaroo_mean.hpp"

#include "../../verify.h"
#include <vector>
#include <algorithm>
using namespace std;

cuckoo_cpu::solver_ctx *cuckoo_ctx = NULL;
cuckaroo_cpu::solver_ctx *cuckaroo_ctx = NULL;

#define HEADERLEN 80

int32_t RunSolverOnCPU(
        uint8_t *header,
        uint64_t nonce,
        uint32_t *result,
        uint32_t resultBuffSize,
        uint32_t *solLength,
        uint32_t *numSol)
{
				using std::vector;
				// printf("[CuckooFind, sols.size()SolutionsCuda] thread: %d\n", getpid());
				//
				if(cuckoo_ctx == NULL && cuckaroo_ctx == NULL){
								printf("the solver context is null\n");
								return 0;
				}

				u32 nsols = 0;
				if(cuckoo_ctx != NULL){
								cuckoo_ctx->setheadernonce((char*)header, HEADERLEN, nonce);
								nsols = cuckoo_ctx->solve();
				}
				else{
								cuckaroo_ctx->setheadernonce((char*)header, HEADERLEN, nonce);
								nsols = cuckaroo_ctx->solve();
				}

				vector<vector<u32> > sols;
				vector<vector<u32> >* solutions = &sols;
				for (unsigned s = 0; s < nsols; s++) {
								u32* prf = NULL;
								if(cuckoo_ctx != NULL) prf = &(cuckoo_ctx->sols[s * PROOFSIZE]);
								else prf = &(cuckaroo_ctx->sols[s * PROOFSIZE]);
								solutions->push_back(vector<u32>());
								vector<u32>& sol = solutions->back();
								for (uint32_t idx = 0; idx < PROOFSIZE; idx++) {
												sol.push_back(prf[idx]);
								}
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


void create_solver_ctx(SolverParams* params, int selected) {
  if (params->nthreads == 0) params->nthreads = 1;
  if (params->ntrims == 0) params->ntrims = EDGEBITS > 30 ? 96 : 68;
  if(selected == 0){
	  cuckoo_ctx = new cuckoo_cpu::solver_ctx(
		 params->nthreads,
		 params->ntrims,
		 params->allrounds,
		 params->showcycle,
		 params->mutate_nonce);

  }else{
	  cuckaroo_ctx = new cuckaroo_cpu::solver_ctx(
		 params->nthreads,
		 params->ntrims,
		 params->allrounds,
		 params->showcycle,
		 params->mutate_nonce);
  }
}

/*void CuckooInitializeCPU(uint32_t* devices, uint32_t deviceNum, int selected = 0, int printDeviceInfo = 1) {
  SolverParams params;
  params.nthreads = deviceNum; //nthreads;
  params.ntrims = 0; //ntrims;
  params.showcycle = true;//showcycle;
  params.allrounds = false;//allrounds;

  create_solver_ctx(&params, selected);
}

int monitor(unsigned int device_count, unsigned int *fanSpeeds, unsigned int *temperatures){
				return 0;
}
void CuckooFinalizeCPU(){
}
void CuckooFinalize(){
}

int32_t CuckooVerifyProof(uint8_t *header, uint64_t nonce, result_t *result) {
    uint64_t littleEndianNonce = htole64(nonce);
    char headerBuf[40];
    memcpy(headerBuf, header, 32);
    memcpy(headerBuf + 32, static_cast<uint64_t*>(&littleEndianNonce), sizeof(nonce));
    siphash_keys keys;
    setheader(headerBuf, 40, &keys);
    int res = cuckoo_verify(result, &keys);
    return res;
}*/

int32_t CuckooVerifyProof_cuckaroo(uint8_t *header, uint64_t nonce, result_t *result) {
    uint64_t littleEndianNonce = htole64(nonce);
    char headerBuf[40];
    memcpy(headerBuf, header, 32);
    memcpy(headerBuf + 32, static_cast<uint64_t*>(&littleEndianNonce), sizeof(nonce));
    siphash_keys keys;
    setheader(headerBuf, 40, &keys);
    int res = cuckaroo_verify(result, keys);
    return res;
}
