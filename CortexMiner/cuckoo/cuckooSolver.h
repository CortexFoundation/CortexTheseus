#include "param.h"
#include <stdint.h>
#include <stdlib.h>
#include <vector>


using namespace std;
typedef uint32_t u32;
class solver_ctx;

struct cuckoo_sol{
    u32 data[PROOFSIZE];
    cuckoo_sol(){}
    cuckoo_sol(u32* src){
        for(int i=0; i<PROOFSIZE; i++)
            data[i] = src[i];
    }
};

class cuckooSolver{
private:
    //init cuckoo context
    u32 nthreads = 1;
    u32 ntrims = EDGEBITS > 30 ? 96 : 68;
    bool allrounds = false;
    bool showcycle = true;
    solver_ctx* solver;
    bool keyed = false;
    u32 numSols;
    vector<cuckoo_sol> sols;

public:
    void loadParam();
    void initSolver();
    void setHeaderNonce(char* header, u32 len, u32 nonce);
    void solve();
    void release();
    u32 getNumSols(){return sols.size();}
    vector<cuckoo_sol>& getSols(){return sols;}
    bool verifySol(u32* sol);
};

