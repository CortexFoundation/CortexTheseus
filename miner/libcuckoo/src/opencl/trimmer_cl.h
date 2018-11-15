#ifndef H_TRIMMER_CL
#define H_TRIMMER_CL
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <stdint.h>
#include <sys/time.h> // gettimeofday
#include <unistd.h>
#include <sys/types.h>
#include <ocl.h>
#include "../cuckoo.h"
namespace cuckoogpu {
// TODO(tian) refactor functions under this namespace

//extern __constant cl_int2 recoveredges[PROOFSIZE];

//__kernel void Recovery(const siphash_keys &sipkeys, __global cl_ulong4 *buffer, __global int *indexes);

typedef uint8_t u8;
typedef uint16_t u16;
#define result_t uint32_t

typedef u32 node_t;
typedef u64 nonce_t;

#ifndef XBITS
#define XBITS ((EDGEBITS-16)/2)
#endif

#define NODEBITS (EDGEBITS + 1)
#define NNODES ((node_t)1 << NODEBITS)
#define NODEMASK (NNODES - 1)

#define IDXSHIFT 10
#define CUCKOO_SIZE (NNODES >> IDXSHIFT)
#define CUCKOO_MASK (CUCKOO_SIZE - 1)
// number of (least significant) key bits that survives leftshift by NODEBITS
#define KEYBITS (64-NODEBITS)
#define KEYMASK ((1L << KEYBITS) - 1)
#define MAXDRIFT (1L << (KEYBITS - IDXSHIFT))

const static u32 MAXEDGES = 0x1000000;

const static u32 NX        = 1 << XBITS;
const static u32 NX2       = NX * NX;
const static u32 XMASK     = NX - 1;
const static u32 X2MASK    = NX2 - 1;
const static u32 YBITS     = XBITS;
const static u32 NY        = 1 << YBITS;
const static u32 YZBITS    = EDGEBITS - XBITS;
const static u32 NYZ       = 1 << YZBITS;
const static u32 ZBITS     = YZBITS - YBITS;
const static u32 NZ        = 1 << ZBITS;

#define EPS_A 133/128
#define EPS_B 85/128

const static u32 ROW_EDGES_A = NYZ * EPS_A;
const static u32 ROW_EDGES_B = NYZ * EPS_B;

const static u32 EDGES_A = ROW_EDGES_A / NX;
const static u32 EDGES_B = ROW_EDGES_B / NX;

#ifndef FLUSHA // should perhaps be in trimparams and passed as template parameter
#define FLUSHA 16
#endif

#ifndef FLUSHB
#define FLUSHB 8
#endif

//template <typename Edge> __kernel bool null(Edge e);

//__kernel node_t dipnode(const siphash_keys &keys, edge_t nce, u32 uorv) ;

//template <typename Edge> u32 __kernel endpoint(const siphash_keys &sipkeys, Edge e, int uorv);

#define checkOpenclErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cl_int code, const char *file, int line, bool abort=true) {
  if (code != CL_SUCCESS) {
    fprintf(stderr, "GPUassert: %s %s %d\n", openclGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


struct blockstpb {
  u16 blocks;
  u16 tpb;
};

struct trimparams {
  u16 expand;
  u16 ntrims;
  blockstpb genA;
  blockstpb genB;
  blockstpb trim;
  blockstpb tail;
  blockstpb recover;

  trimparams() {
    expand              =    0;
    ntrims              =  176;
    genA.blocks         = 4096;
    genA.tpb            =  256;
    genB.blocks         =  NX2;
    genB.tpb            =  128;
    trim.blocks         =  NX2;//NX2;
    trim.tpb            =  256;//512;
    tail.blocks         =  NX2;
    tail.tpb            = 256;//1024;
    recover.blocks      = 1024;//1024;
    recover.tpb         = 256;//1024;
  }
};

typedef u32 proof[PROOFSIZE];

// maintains set of trimmable edges
struct edgetrimmer {
  cl_platform_id platformId;
  cl_device_id deviceId;
  cl_context context;
  cl_command_queue commandQueue;
  cl_program program;
  trimparams tp;
  edgetrimmer *dt;
  size_t sizeA, sizeB;
  size_t indexesSize;
  cl_mem bufferA;
  cl_mem bufferB;
  cl_mem bufferAB;
  cl_mem indexesE;
  cl_mem indexesE2;
  cl_mem recoveredges; //const
  u32 hostA[NX * NY];
  u32 *uvnodes;
  proof sol;
  siphash_keys sipkeys;//, *dipkeys;
  cl_mem dipkeys;

  edgetrimmer(const trimparams _tp, cl_context context, cl_command_queue commandQueue, cl_program program);

  u64 globalbytes() const ;

  ~edgetrimmer();

  u32 trim(uint32_t device);
};

};


#endif
