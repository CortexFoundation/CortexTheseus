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
//    tp.genA.tpb = 128;
    trimmer = new edgetrimmer(tp, _device, 1);
    edges   = new uint2[MAXEDGES];
    device = _device;
    cg = new graph<edge_t>(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT);
  }
  void init(trimparams tp, uint32_t _device = 0) {
    //tp.genA.tpb = 128;
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
  }

  ~cuckaroo_solver_ctx() {
    delete[] edges;
    delete trimmer;
    delete cg;
  }

  int findcycles(u32 nedges) {
    sols.clear();
    cg->reset();
    for (u32 i = 0; i < nedges; i++){
      if(cg->add_compress_edge(edges[i].x, edges[i].y) == -1) {
          printf("add edge failed .........\n");
          return 0;
      }
    }
    for (u32 s = 0 ;s < cg->nsols; s++) {
//       print_log("Solution");
      for (u32 j = 0; j < PROOFSIZE; j++) {
        soledges[j] = edges[cg->sols[s][j]];
//        print_log(" (%u, %u)", soledges[j].x, soledges[j].y);
      }
//       print_log("\n");
      sols.resize(sols.size() + PROOFSIZE);
      checkCudaErrors(cudaMemcpyToSymbol(recoveredges, soledges, sizeof(soledges)));
//      cudaMemset(trimmer->indexesE[1], 0, trimmer->indexesSize);
      Cuckaroo_Recovery<<<trimmer->tp.recover.blocks, trimmer->tp.recover.tpb>>>(*trimmer->dipkeys2, (int *)trimmer->uvnodes);
      checkCudaErrors(cudaMemcpy(&sols[sols.size()-PROOFSIZE], trimmer->uvnodes, PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaDeviceSynchronize());
      qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), cg->nonce_cmp);
//      printf("findcycles:\n");
//      for(int i = 0; i < PROOFSIZE; i++){
//        printf("%u, ", sols[sols.size()-PROOFSIZE + i]);
//      }
//      printf("\n");
    }
    return sols.size() / PROOFSIZE;
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
    cudaMemcpy(edges, trimmer->bufferB, nedges * sizeof(uint2), cudaMemcpyDeviceToHost);
    // gettimeofday(&time1, 0);
    // timems = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    // gettimeofday(&time0, 0);
//    findcycles(nedges);
    // gettimeofday(&time1, 0);
    // timems2 = (time1.tv_sec-time0.tv_sec)*1000 + (time1.tv_usec-time0.tv_usec)/1000;
    // printf("findcycles edges %d time %d ms total %d ms\n", nedges, timems2, timems+timems2);
  //  return sols.size() / PROOFSIZE;
  	checkCudaErrors(cudaMemcpy(trimmer->dipkeys2, trimmer->dipkeys, sizeof(siphash_keys), cudaMemcpyDeviceToDevice));
  	return nedges;
  }
};

}; // end of namespace cuckoogpu

