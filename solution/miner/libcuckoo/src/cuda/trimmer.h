#ifndef H_TRIMMER
#define H_TRIMMER
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <stdint.h>
#include <sys/time.h> // gettimeofday
#include <unistd.h>
#include <sys/types.h>
#include "../cuckoo.h"
namespace cuckoogpu {
// TODO(tian) refactor functions under this namespace

extern __constant__ uint2 recoveredges[PROOFSIZE];

__global__ void Cuckoo_Recovery(const siphash_keys &sipkeys, int *indexes);

__global__ void Cuckaroo_Recovery(const siphash_keys &sipkeys, int *indexes);

typedef uint8_t u8;
typedef uint16_t u16;
#define result_t uint32_t

typedef u32 node_t;
typedef u64 nonce_t;

#ifndef XBITS
//#define XBITS ((EDGEBITS-16)/2)
#define XBITS 6
#endif
#ifndef YBITS
#define YBITS 7
#endif

#define NODEBITS (EDGEBITS + 1)
#define NNODES ((node_t)1 << NODEBITS)
#define NODEMASK (NNODES - 1)

#define IDXSHIFT 12
#define CUCKOO_SIZE (NNODES >> IDXSHIFT)
#define CUCKOO_MASK (CUCKOO_SIZE - 1)
// number of (least significant) key bits that survives leftshift by NODEBITS
#define KEYBITS (64-NODEBITS)
#define KEYMASK ((1L << KEYBITS) - 1)
#define MAXDRIFT (1L << (KEYBITS - IDXSHIFT))

const static u32 MAXEDGES = 0x1000000;

const static u32 NX        = 1 << XBITS;
const static u32 NY        = 1 << YBITS;
const static u32 NX2       = NX * NY;
const static u32 XMASK     = NX - 1;
const static u32 YMASK     = NY - 1;
const static u32 X2MASK    = NX2 - 1;
//const static u32 YBITS     = XBITS;
const static u32 YZBITS    = EDGEBITS - XBITS;
const static u32 NYZ       = 1 << YZBITS;
const static u32 ZBITS     = YZBITS - YBITS;
const static u32 NZ        = 1 << ZBITS;
const u32 ZMASK     = NZ - 1;

#ifndef NEPS_A
#define NEPS_A 128
#endif
#ifndef NEPS_B
#define NEPS_B 80
#endif
#define NEPS 128

const u32 EDGES_A = NZ * NEPS_A / NEPS;
const u32 EDGES_B = NZ * NEPS_B / NEPS;

const u32 ROW_EDGES_A = EDGES_A * NY;
const u32 ROW_EDGES_B = EDGES_B * NY;

// Number of Parts of BufferB, all but one of which will overlap BufferA
#ifndef NRB1
#define NRB1 (NX / 2)
#endif
#define NRB2 (NX - NRB1)
#ifndef NB
#define NB 2
#endif

#ifndef NA
#define NA  ((NB * NEPS_A + NEPS_B-1) / NEPS_B)
#endif

#ifndef FLUSHA // should perhaps be in trimparams and passed as template parameter
#define FLUSHA 16
#endif

#ifndef FLUSHB
#define FLUSHB 8
#endif

template <typename Edge> __device__ bool null(Edge e);

__device__ node_t dipnode(const siphash_keys &keys, edge_t nce, u32 uorv) ;

template <typename Edge> u32 __device__ endpoint(const siphash_keys &sipkeys, Edge e, int uorv);

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	int device;
	cudaGetDevice(&device);
  if (code != cudaSuccess) {
    fprintf(stderr, "the GPU #%d assert: %s\n", device, cudaGetErrorString(code));
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
    genA.blocks         = NX2;
    genA.tpb            =  256;
    genB.blocks         =  NX2;
    genB.tpb            =  128;
    trim.blocks         =  NX2;
    trim.tpb            =  512;
    tail.blocks         =  NX2;
    tail.tpb            = 1024;
    recover.blocks      = 1024;
    recover.tpb         = 1024;
  }
};

typedef u32 proof[PROOFSIZE];

// maintains set of trimmable edges
struct edgetrimmer {
  trimparams tp;
  edgetrimmer *dt;
  u32 deviceId;
  size_t sizeA, sizeB;
  size_t indexesSize;
  uint8_t *bufferA;
  uint8_t *bufferB;
  uint8_t *bufferAB;
  u32 *indexesE[1+NB];
  u32 nedges;
  u32 *uvnodes;
  proof sol;
  int selected;
  siphash_keys sipkeys, *dipkeys, *dipkeys2;

  edgetrimmer(const trimparams _tp, u32 _deviceId, int _selected);

  u64 globalbytes() const ;

  ~edgetrimmer();

  u32 trim(uint32_t device);
};

};


#endif
