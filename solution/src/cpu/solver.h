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
namespace cuckoogpu { 

int nonce_cmp(const void *a, const void *b) {
  return *(u32 *)a - *(u32 *)b;
}
typedef u32 proof[PROOFSIZE];

struct solver_ctx {
  std::vector<u32> sols; // concatanation of all proof's indices
  virtual void setheadernonce(char* const headernonce, const u32 len, const u32 nonce) = 0;
  virtual int solve() = 0;
  virtual void findcycles() = 0;

  solver_ctx(){}
  solver_ctx(const u32 n_threads, const u32 n_trims, bool allrounds, bool show_cycle, bool mutate_nonce) {}
  virtual ~solver_ctx(){}
};

}; // end of namespace cuckoogpu

#endif
