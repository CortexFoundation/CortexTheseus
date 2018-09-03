#include "CuckooSolver.h"
#include "mean_miner_new.hpp"

#include <thread>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

CuckooSolver::CuckooSolver() {
    _run = false;
}

void CuckooSolver::loadParam(){
}

void CuckooSolver::initSolver(){
    _run = true;
    printf("Initializing Cuckoo Cycle Solver...\n");
    // solver_ctx ctx(nthreads, ntrims, allrounds, showcycle);

    solver = new solver_ctx(nthreads, ntrims, allrounds, showcycle);
    u64 sbytes = solver->sharedbytes();
    u32 tbytes = solver->threadbytes();
    int sunit,tunit;
    for (sunit=0; sbytes >= 10240; sbytes>>=10,sunit++) ;
    for (tunit=0; tbytes >= 10240; tbytes>>=10,tunit++) ;
    // printf("Using %d%cB bucket memory at %lx,\n", sbytes, " KMGT"[sunit], (u64)ctx.trimmer->buckets);
    // printf("%dx%d%cB thread memory at %lx,\n", nthreads, tbytes, " KMGT"[tunit], (u64)ctx.trimmer->tbuckets);
    // printf("%d-way siphash, and %d buckets.\n", NSIPHASH, NX);

    sols.clear();
    keyed = false;
    bHashVerify = false;
    numSols = 0;
    printf("Initialized completed!\n");
    _run = false;
}

void CuckooSolver::stop() {
    solver->_stop = true;
}

void CuckooSolver::await() {
    while (_run);
    printf("ThreadID : %lu, Solver: %p, stoped\n", std::this_thread::get_id(), (void*)solver);
}

void CuckooSolver::setHeaderNonce(char* header, u32 len, u32 nonce){
    if(header==NULL){
        header = new char[HEADERLEN];
        memset(header, 0, sizeof(char)*HEADERLEN);
        solver->setheadernonce(header, HEADERLEN, nonce);
        delete header;
    }
    else
        solver->setheadernonce(header, len, nonce);

    sols.clear();
    keyed = true;
}

void CuckooSolver::solve(){
    if (solver->_stop) return ;

    _run = true;
    u32 nsols = solver->solve();
    u32 sumnsols = 0;
    //gettimeofday(&time1, 0);
    //timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    //printf("Time: %d ms\n", timems);

    // 2 verify solutions
    for (unsigned s = 0; s < nsols; s++) {

        // printf("Solution");
        u32* prf = & (solver->sols[s * PROOFSIZE]);
        /*
        for (u32 i = 0; i < PROOFSIZE; i++)
            printf(" %jx", (uintmax_t)prf[i]);
        printf("\n");
        */

        if(verifySol(prf)){
            printf("valid solution found.\n");
            sumnsols++;

            unsigned char cyclehash[32];
            blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)prf, sizeof(proof), 0, 0);
            cuckoo_sol s(prf, cyclehash);
            sols.push_back(s);
        }

    }
    //sumnsols += nsols;
    numSols = sumnsols;
    // printf("%d total solutions %d\n", sumnsols, sols.size());

    _run = false;
}

void CuckooSolver::release(){
    delete solver;
}

bool CuckooSolver::verifySol(u32* sol){
    // make sure cuckoo solver is initialized with correct header & nonce
    if(!keyed){
        printf("error: cuckoo solver header nonce invalid\n");
        return false;
    }
    int pow_rc = verify(sol, &(solver->trimmer->sip_keys));
    if(pow_rc != POW_OK){
        printf("FAILED due to %s\n", errstr[pow_rc]);
        return false;
    }


    if(bHashVerify){
        bool valid = true;
        printf("Verified with cyclehash ");
        unsigned char cyclehash[32];
        blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)sol, sizeof(proof), 0, 0);
        for (int i=0; i<32; i++){
            printf("%02x", cyclehash[i]);
        }
        printf("\n");
        for(int i=0; i<32; i++){
        printf("%02x %02x\n",cyclehash[i],target[i]);
            if(cyclehash[i] >  target[i]){
                printf("difficulty is not satisfied\n");
                valid = false;
                break;
            } else if(cyclehash[i]<target[i]){
                valid = true;
                break;
            }
        }
        return valid;
    }

    return true;
}

bool CuckooSolver::verifySol(u32* sol, uchar* hash, uchar* target){
    // make sure cuckoo solver is initialized with correct header & nonce
    if(!keyed){
        printf("error: cuckoo solver header nonce invalid\n");
        return false;
    }
    int pow_rc = verify(sol, &(solver->trimmer->sip_keys));
    if(pow_rc != POW_OK){
        printf("FAILED due to %s\n", errstr[pow_rc]);
        return false;
    }

    bool valid = true;
    printf("Verified with cyclehash ");
    unsigned char cyclehash[32];
    blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)sol, sizeof(proof), 0, 0);
    /*for (int i=0; i<32; i++){
        printf("%02x", cyclehash[i]);
    }
    printf("\n");*/

    for(int i=0; i<32; i++){
        if(cyclehash[i] != hash[i]){
            printf("hash mismatch error\n");
            valid = false;
            return valid;
        }
    }
    for (int i = 0;i<32;i++){
        printf("%02x %02x\n",cyclehash[i],target[i]);
        if(cyclehash[i] >  target[i]){
            printf("difficulty is not satisfied\n");
            valid = false;
            return valid;
        }else if(cyclehash[i]<target[i]){
                valid = true;
                break;
            }
    }
    return valid;
}


void CuckooSolver::setHashTarget(unsigned char* target_){
    // make sure target_ is 32-byte long
    for(int i=0; i<32; i++)
        target[i] = target_[i];
    bHashVerify = true;
}
