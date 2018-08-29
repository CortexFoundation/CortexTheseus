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
    void start();
    // void testEquihash();
    void testCuckoo();

    void CuckooInit(int thread = 1);
    bool CuckooSolve(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result, uint32_t *result_len,uchar* target,uchar* result_hash);
    bool CuckooVerify(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result, uchar* target, uchar* hash);
    void CuckooRelease();
};

