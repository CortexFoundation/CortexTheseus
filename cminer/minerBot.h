#pragma once

#define __QNXNTO__
#include "cuckoo/param.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
//#include "./util/XMLhelper.h"
#include "cuckoo/CuckooSolver.h"


class MinerBot
{
  private:
    int mode;
    CuckooSolver cs;

  public:
    pthread_mutex_t mutex;

    MinerBot(unsigned int nthread);
    virtual ~MinerBot();
    void loadParam();
	void stop();
    void await();
    // void testEquihash();

    CuckooSolver* GetSolver();

    void CuckooInit(int thread = 1);
    bool CuckooSolve(char *header, uint32_t header_len, uint64_t nonce, uint32_t *result, uint32_t *result_len,uchar* target,uchar* result_hash);
    bool CuckooSolve(const uint8_t *header, uint32_t headerLength, uint64_t nonce, vector<vector<uint32_t>>* solutions);
    bool CuckooVerify(char *header, uint32_t header_len, uint64_t nonce, uint32_t *result, uchar* target, uchar* hash);
    bool CuckooVerify_cuckaroo(char *header, uint32_t header_len, uint64_t nonce, uint32_t *result, uchar* target, uchar* hash);
};

