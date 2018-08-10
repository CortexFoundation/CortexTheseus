#include "param.h"
#include <stdint.h>
#include <stdlib.h>
#include <vector>


using namespace std;
typedef uint32_t u32;
typedef uint8_t uchar;


struct cuckoo_sol{
    u32 data[PROOFSIZE];
    unsigned char hash[32];
    cuckoo_sol(){}
    cuckoo_sol(u32* src, uchar* hash_){
        for(int i=0; i<PROOFSIZE; i++)
            data[i] = src[i];
        for(int i=0; i<32; i++){
            hash[i] = hash_[i];
        }
    }
};



// mean miner

class solver_ctx;

class cuckooSolver{
private:
    // cuckoo context
    u32 nthreads = 1;
    u32 ntrims = EDGEBITS > 30 ? 96 : 68;
    bool allrounds = false;
    bool showcycle = true;
    solver_ctx* solver;
    
    
    bool keyed = false;     // if siphash key is set
    bool bHashVerify = false; // if hash test is needed

    u32 numSols;
    vector<cuckoo_sol> sols;
    unsigned char target[32];

public:
    void loadParam();
    void initSolver();
    void release();

    // set input, and solve
    void setHeaderNonce(char* header, u32 len, u32 nonce);
    void solve();
    
    // get solutions and verify
    u32 getNumSols(){return sols.size();}
    vector<cuckoo_sol>& getSols(){return sols;}
    bool verifySol(u32* sol);
    bool verifySol(u32* sol, uchar* hash, uchar* target);

    // adjust difficulty
    void setHashTarget(unsigned char* target_);
};
