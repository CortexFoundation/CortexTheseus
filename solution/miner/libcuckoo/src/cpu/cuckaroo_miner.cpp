#include "cuckaroo_mean.hpp"
#include "../../verify.h"
#include <vector>
#include <algorithm>
using namespace std;

typedef solver_ctx SolverCtx;
SolverCtx *ctx;


#define HEADERLEN 80
int32_t RunSolverOnCPU(
        uint8_t *header,
        uint64_t nonce,
    		uint32_t threadId,
        uint32_t *result,
        uint32_t resultBuffSize,
        uint32_t *solLength,
        uint32_t *numSol)
{
    using std::vector;
    // printf("[CuckooFind, sols.size()SolutionsCuda] thread: %d\n", getpid());
		ctx->setheadernonce((char*)header, HEADERLEN, nonce);
		u32 nsols = ctx->solve();
		
    vector<vector<u32> > sols;
    vector<vector<u32> >* solutions = &sols;
    for (unsigned s = 0; s < nsols; s++) {
        u32* prf = &(ctx->sols[s * PROOFSIZE]);
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


SolverCtx* create_solver_ctx(SolverParams* params, int selected) {
  if (params->nthreads == 0) params->nthreads = 1;
  if (params->ntrims == 0) params->ntrims = EDGEBITS > 30 ? 96 : 68;
  SolverCtx* ctx = new solver_ctx(
		 params->nthreads,
		 params->ntrims,
		 params->allrounds,
		 params->showcycle,
		 params->mutate_nonce);
  return ctx;
}

void CuckooInitialize(uint32_t* devices, uint32_t deviceNum, int selected = 0) {
  SolverParams params;
  params.nthreads = 0; //nthreads;
  params.ntrims = 0; //ntrims;
  params.showcycle = false;//showcycle;
  params.allrounds = false;//allrounds;

  SolverCtx* ctx = create_solver_ctx(&params, selected);
}

int32_t FindSolutionsByGPU(
        uint8_t *header,
        uint64_t nonce,
    uint32_t threadId
    )
{
				return 0;
}

int32_t FindCycles(
	uint32_t threadId, 
	uint32_t nedges,
	uint32_t *result,
	uint32_t resultBuffSize,
	uint32_t *solLength,
	uint32_t *numSol){
				return 0;
}

int monitor(unsigned int device_count, unsigned int *fanSpeeds, unsigned int *temperatures){
				return 0;
}
