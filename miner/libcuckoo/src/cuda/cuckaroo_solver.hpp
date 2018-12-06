// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license
#include "solver.h"
#include "graph.hpp"

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

namespace cuckoogpu { 

struct cuckaroo_solver_ctx : public solver_ctx{
  uint2 *edges;
  graph<edge_t> *cg;
  uint2 soledges[PROOFSIZE];

  cuckaroo_solver_ctx(){}
  cuckaroo_solver_ctx(trimparams tp, uint32_t _device = 0) {
    tp.genA.tpb = 128;
    trimmer = new edgetrimmer(tp, _device, 1);
    edges   = new uint2[MAXEDGES];
    device = _device;
    cg = new graph<edge_t>(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT);
  }
  void init(trimparams tp, uint32_t _device = 0) {
    tp.genA.tpb = 128;
    trimmer = new edgetrimmer(tp, _device, 1);
    edges   = new uint2[MAXEDGES];
    device = _device;
    cg = new graph<edge_t>(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT);
  }
  
  void setheadernonce(char * const header,  const uint64_t nonce) {
    uint64_t littleEndianNonce = htole64(nonce);
    char headerBuf[40];
    memcpy(headerBuf, header, 32);
    memcpy(headerBuf + 32, static_cast<uint64_t*>(&littleEndianNonce), sizeof(nonce));
    setheader(headerBuf, 40, &trimmer->sipkeys);
    sols.clear();
  }

  ~cuckaroo_solver_ctx() {
    delete[] edges;
    delete trimmer;
    delete cg;
  }

  int findcycles(uint2 *edges, u32 nedges) {
    cg->reset();
    for (u32 i = 0; i < nedges; i++){
      cg->add_compress_edge(edges[i].x, edges[i].y);
    }
    for (u32 s = 0 ;s < cg->nsols; s++) {
      // print_log("Solution");
      for (u32 j = 0; j < PROOFSIZE; j++) {
        soledges[j] = edges[cg->sols[s][j]];
        // print_log(" (%x, %x)", soledges[j].x, soledges[j].y);
      }
      // print_log("\n");
      sols.resize(sols.size() + PROOFSIZE);
      cudaMemcpyToSymbol(recoveredges, soledges, sizeof(soledges));
      cudaMemset(trimmer->indexesE2, 0, trimmer->indexesSize);
      Cuckaroo_Recovery<<<trimmer->tp.recover.blocks, trimmer->tp.recover.tpb>>>(*trimmer->dipkeys, trimmer->bufferA, (int *)trimmer->indexesE2);
      cudaMemcpy(&sols[sols.size()-PROOFSIZE], trimmer->indexesE2, PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);
      checkCudaErrors(cudaDeviceSynchronize());
      qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), cg->nonce_cmp);
    }
    return 0;
  }
  int solve() {
    // u32 timems,timems2;
    // struct timeval time0, time1;
//	printf("call cuckaroo...\n");
    // gettimeofday(&time0, 0);
    u32 nedges = trimmer->trim(this->device);
    //printf("trim result: %u\n", nedges);
    if (nedges > MAXEDGES) {
      fprintf(stderr, "OOPS; losing %d edges beyond MAXEDGES=%d\n", nedges-MAXEDGES, MAXEDGES);
      nedges = MAXEDGES;
    }
	// nedges must less then CUCKOO_SIZE, or find-cycle procedure will never stop.
	nedges = nedges & CUCKOO_MASK;
    cudaMemcpy(edges, trimmer->bufferB, nedges * 8, cudaMemcpyDeviceToHost);
    // gettimeofday(&time1, 0);
    // timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    // gettimeofday(&time0, 0);
    findcycles(edges, nedges);
    // gettimeofday(&time1, 0);
    // timems2 = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    // printf("findcycles edges %d time %d ms total %d ms\n", nedges, timems2, timems+timems2);
    return sols.size() / PROOFSIZE;
  }
};

}; // end of namespace cuckoogpu

