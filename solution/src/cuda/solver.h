#ifndef SOLVER_H
#define SOLVER_H

#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <vector>
// #include <algorithm>
#include <stdint.h>
#include <sys/time.h> // gettimeofday
#include <unistd.h>
#include <sys/types.h>
#include "trimmer.h"
namespace cuckoogpu { 

struct solver_ctx {
  edgetrimmer *trimmer;
  std::vector<u32> sols; // concatenation of all proof's indices
  uint32_t device;

  virtual void init(trimparams tp, uint32_t _device = 0) = 0;
  virtual void setheadernonce(char * const header,  const uint64_t nonce) = 0;
  virtual int solve() = 0;
  virtual int findcycles(u32 nedges) = 0;

  solver_ctx(){}
  virtual ~solver_ctx(){}
};

}; // end of namespace cuckoogpu

#endif
