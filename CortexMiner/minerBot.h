

#pragma once

#define __QNXNTO__
#include <stdio.h>
#include <stdlib.h>
//#include "./util/XMLhelper.h"
#include "cuckoo/cuckooSolver.h"


class minerBot
{
  private:
    int mode;
    cuckooSolver *cs;

  public:
    void loadParam();
    void start();
    // void testEquihash();
    void testCuckoo();

    void CuckooInit();
    void CuckooSolve(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result, uint32_t *result_len,uchar* result_hash,uchar* target);
    bool CuckooVerify(char *header, uint32_t header_len, uint32_t nonce, uint32_t *result, uchar* hash, uchar* target);
    void CuckooRelease();
};
