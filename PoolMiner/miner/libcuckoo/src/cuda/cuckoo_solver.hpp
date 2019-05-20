// Cuckoo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license
#include "solver.h"

namespace cuckoogpu { 

class cuckoo_hash {
public:
  u64 *cuckoo;

  cuckoo_hash() {
    cuckoo = new u64[CUCKOO_SIZE];
	memset(cuckoo, 0, CUCKOO_SIZE * sizeof(u64));
  }
  ~cuckoo_hash() {
    delete[] cuckoo;
  }
  void set(node_t u, node_t v) {
    u64 niew = (u64)u << NODEBITS | v;
    for (node_t ui = (u >> IDXSHIFT) & CUCKOO_MASK; ; ui = (ui+1) & CUCKOO_MASK) {
      u64 old = cuckoo[ui];
      if (old == 0 || (old >> NODEBITS) == (u & KEYMASK)) {
        cuckoo[ui] = niew;
        return;
      }
    }
  }
  node_t operator[](node_t u) const {
    for (node_t ui = (u >> IDXSHIFT) & CUCKOO_MASK; ; ui = (ui+1) & CUCKOO_MASK) {
      u64 cu = cuckoo[ui];
      if (!cu)
        return 0;
      if ((cu >> NODEBITS) == (u & KEYMASK)) {
        assert(((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
        return (node_t)(cu & NODEMASK);
      }
    }

  }
};

const static u32 MAXPATHLEN = 8 << ((NODEBITS+2)/3);

int nonce_cmp(const void *a, const void *b) {
  return *(u32 *)a - *(u32 *)b;
}

struct cuckoo_solver_ctx : public solver_ctx{
  uint2 *edges;
  cuckoo_hash *cuckoo;
  uint2 soledges[PROOFSIZE];
  u32 us[MAXPATHLEN];
  u32 vs[MAXPATHLEN];

  cuckoo_solver_ctx(){}
  cuckoo_solver_ctx(const trimparams tp, uint32_t _device = 0) {
    trimmer = new edgetrimmer(tp, _device, 0);
    edges   = new uint2[MAXEDGES];
    cuckoo  = new cuckoo_hash();
    device = _device;
  }
  void init(const trimparams tp, uint32_t _device = 0) {
    trimmer = new edgetrimmer(tp, _device, 0);
    edges   = new uint2[MAXEDGES];
    cuckoo  = new cuckoo_hash();
    device = _device;
  }
  
  void setheadernonce(char * const header,  const uint64_t nonce) {
    uint64_t littleEndianNonce = htole64(nonce);
    char headerBuf[40];
    memcpy(headerBuf, header, 32);
    memcpy(headerBuf + 32, static_cast<uint64_t*>(&littleEndianNonce), sizeof(nonce));
    setheader(headerBuf, 40, &trimmer->sipkeys);
  }

  ~cuckoo_solver_ctx() {
    delete cuckoo;
    delete[] edges;
    delete trimmer;
  }

  void recordedge(const u32 i, const u32 u2, const u32 v2) {
    soledges[i].x = u2/2;
    soledges[i].y = v2/2;
  }

  void solution(const u32 *us, u32 nu, const u32 *vs, u32 nv) {
    u32 ni = 0;
    recordedge(ni++, *us, *vs);
    while (nu--)
      recordedge(ni++, us[(nu+1)&~1], us[nu|1]); // u's in even position; v's in odd
    while (nv--)
      recordedge(ni++, vs[nv|1], vs[(nv+1)&~1]); // u's in odd position; v's in even
    assert(ni == PROOFSIZE);
    sols.resize(sols.size() + PROOFSIZE);
    cudaMemcpyToSymbol(recoveredges, soledges, sizeof(soledges));
//    cudaMemset(trimmer->indexesE[0], 0, trimmer->indexesSize);
	Cuckoo_Recovery<<<trimmer->tp.recover.blocks, trimmer->tp.recover.tpb>>>(*trimmer->dipkeys2, (int *)trimmer->uvnodes);
    cudaMemcpy(&sols[sols.size()-PROOFSIZE], trimmer->uvnodes, PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());
/*	fprintf(stderr, "Index: %zu points: [", sols.size() / PROOFSIZE);
	for (uint32_t idx = 0; idx < PROOFSIZE; idx++) {
		fprintf(stderr, "<%u,%u>, ", soledges[idx].x, soledges[idx].y);
	}
	fprintf(stderr, "] solutions: [");
	for (uint32_t idx = 0; idx < PROOFSIZE; idx++) {
		fprintf(stderr, "%u,", sols[sols.size() - PROOFSIZE + idx]);
	}
	fprintf(stderr, "]\n");
	*/
    qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), nonce_cmp);
  }

  u32 path(u32 u, u32 *us) {
    u32 nu, u0 = u;
	/* fprintf(stderr, "start %zu\n", u0); */
    for (nu = 0; u; u = (*cuckoo)[u]) {
      if (nu >= MAXPATHLEN) {
			/* fprintf(stderr, "nu: %zu, u: %zu, Maxpathlen: %zu\n", nu, u, MAXPATHLEN); */
        while (nu-- && us[nu] != u) ;
        if (~nu) {
          printf("illegal %4d-cycle from node %d\n", MAXPATHLEN-nu, u0);
          exit(0);
        }
        printf("maximum path length exceeded\n");
        return 0; // happens once in a million runs or so; signal trouble
      }
      us[nu++] = u;
    }
	/* fprintf(stderr, "path nu: %zu\n", nu); */
    return nu;
  }

  void addedge(uint2 edge) {
    const u32 u0 = edge.x << 1, v0 = (edge.y << 1) | 1;

    if (u0) {
      u32 nu = path(u0, us), nv = path(v0, vs);
      if (!nu-- || !nv--)
        return; // drop edge causing trouble

      if (us[nu] == vs[nv]) {
        const u32 min = nu < nv ? nu : nv;
        for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
        const u32 len = nu + nv + 1;
        if (len == PROOFSIZE)
          solution(us, nu, vs, nv);
        // if (len == 2) printf("edge %x %x\n", edge.x, edge.y);
      } else if (nu < nv) {
        while (nu--)
          cuckoo->set(us[nu+1], us[nu]);
        cuckoo->set(u0, v0);
      } else {
        while (nv--)
          cuckoo->set(vs[nv+1], vs[nv]);
		cuckoo->set(v0, u0);
      }

    }
  }

  int findcycles(u32 nedges) {
    sols.clear();
    memset(cuckoo->cuckoo, 0, CUCKOO_SIZE * sizeof(u64));
    for (u32 i = 0; i < nedges; i++) {
      addedge(edges[i]);
    }
    return sols.size() / PROOFSIZE;
  }

  int solve() {
    // u32 timems,timems2;
    // struct timeval time0, time1;

    // gettimeofday(&time0, 0);
    u32 nedges = trimmer->trim(this->device);
//    printf("trim result: %u\n", nedges);
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


