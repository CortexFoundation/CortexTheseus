// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp

#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset
#include <stdint.h>    // for types uint32_t,uint64_t
#include <immintrin.h> // for _mm256_* intrinsics
#ifndef __APPLE__
#include <endian.h>    // for htole32/64
#else
#include <machine/endian.h>
#include <libkern/OSByteOrder.h>
#define htole32(x) OSSwapHostToLittleInt32(x)
#define htole64(x) OSSwapHostToLittleInt64(x)
#endif

#include "param.h"
#include "blake2.h"

// save some keystrokes since i'm a lazy typer
typedef uint32_t u32;
typedef uint64_t u64;

// siphash uses a pair of 64-bit keys,
typedef struct {
  u64 k0;
  u64 k1;
  u64 k2;
  u64 k3;
} siphash_keys;

#define U8TO64_LE(p) ((p))

// set doubled (128->256 bits) siphash keys from 32 byte char array
void setkeys(siphash_keys *keys, const char *keybuf);

#define ROTL(x,b) (u64)( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
  do { \
    v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
    v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
    v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
    v1 = ROTL(v1,17);   v3 = ROTL(v3,21); \
    v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
  } while(0)

// SipHash-2-4 without standard IV xor and specialized to precomputed key and 8 byte nonces
u64 siphash24(const siphash_keys *keys, const u64 nonce);
// standard siphash24 definition can be recovered by calling setkeys with
// k0 ^ 0x736f6d6570736575ULL, k1 ^ 0x646f72616e646f6dULL,
// k2 ^ 0x6c7967656e657261ULL, and k1 ^ 0x7465646279746573ULL

void setkeys(siphash_keys *keys, const char *keybuf) {
  keys->k0 = htole64(((u64 *)keybuf)[0]);
  keys->k1 = htole64(((u64 *)keybuf)[1]);
  keys->k2 = htole64(((u64 *)keybuf)[2]);
  keys->k3 = htole64(((u64 *)keybuf)[3]);
}

u64 siphash24(const siphash_keys *keys, const u64 nonce) {
  u64 v0 = keys->k0, v1 = keys->k1, v2 = keys->k2, v3 = keys->k3 ^ nonce;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1) ^ (v2  ^ v3);
}

#ifdef SIPHASH_COMPAT
#include <stdio.h>
#endif

// proof-of-work parameters
#ifndef EDGEBITS
// the main parameter is the 2-log of the graph size,
// which is the size in bits of the node identifiers
#define EDGEBITS 28
#endif
#ifndef PROOFSIZE
// the next most important parameter is the (even) length
// of the cycle to be found. a minimum of 12 is recommended
#define PROOFSIZE
#endif

#if EDGEBITS > 32
typedef u64 edge_t;
#else
typedef u32 edge_t;
#endif
#if EDGEBITS > 31
typedef u64 node_t;
#else
typedef u32 node_t;
#endif

// number of edges
#define NEDGES ((node_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK ((edge_t)NEDGES - 1)

// generate edge endpoint in cuckoo graph without partition bit
node_t sipnode(siphash_keys *keys, edge_t edge, u32 uorv) {
  return siphash24(keys, 2*edge + uorv) & EDGEMASK;
}

enum verify_code { POW_OK, POW_HEADER_LENGTH, POW_TOO_BIG, POW_TOO_SMALL, POW_NON_MATCHING, POW_BRANCH, POW_DEAD_END, POW_SHORT_CYCLE};
const char *errstr[] = { "OK", "wrong header length", "edge too big", "edges not ascending", "endpoints don't match up", "branch in cycle", "cycle dead ends", "cycle too short"};

// verify that edges are ascending and form a cycle in header-generated graph
int verify(edge_t edges[PROOFSIZE], siphash_keys *keys) {
  printf("edges: \n");
  for (uint32_t i = 0; i < PROOFSIZE; i++) {
      printf(" %d", edges[i]);
  }
  printf("\n");
  printf("sig: \n");
  printf(" %d %d %d %d", keys->k0, keys->k1, keys->k2, keys->k3);
  printf("\n");
  node_t uvs[2*PROOFSIZE];
  node_t xor0 = 0, xor1  =0;
  for (u32 n = 0; n < PROOFSIZE; n++) {
    if (edges[n] > EDGEMASK)
      return POW_TOO_BIG;
    if (n && edges[n] <= edges[n-1])
      return POW_TOO_SMALL;
    xor0 ^= uvs[2*n  ] = sipnode(keys, edges[n], 0);
    xor1 ^= uvs[2*n+1] = sipnode(keys, edges[n], 1);
  }
  

  if (xor0|xor1)              // optional check for obviously bad proofs
    return POW_NON_MATCHING;

  u32 n = 0, i = 0, j;
  do {                        // follow cycle
    for (u32 k = j = i; (k = (k+2) % (2*PROOFSIZE)) != i; ) {
      if (uvs[k] == uvs[i]) { // find other edge endpoint identical to one at i
        if (j != i)           // already found one before
          return POW_BRANCH;
        j = k;
      }
    }
    if (j == i) return POW_DEAD_END;  // no matching endpoint
    i = j^1;
    n++;
  } while (i != 0);           // must cycle back to start or we would have found branch
  return n == PROOFSIZE ? POW_OK : POW_SHORT_CYCLE;
}

// convenience function for extracting siphash keys from header
void setheader(const char *header, const u32 headerlen, siphash_keys *keys) {
  char hdrkey[32];
  // SHA256((unsigned char *)header, headerlen, (unsigned char *)hdrkey);
  blake2b((void *)hdrkey, sizeof(hdrkey), (const void *)header, headerlen, 0, 0);
#ifdef SIPHASH_COMPAT
  u64 *k = (u64 *)hdrkey;
  u64 k0 = k[0];
  u64 k1 = k[1];
  printf("k0 k1 %lx %lx\n", k0, k1);
  k[0] = k0 ^ 0x736f6d6570736575ULL;
  k[1] = k1 ^ 0x646f72616e646f6dULL;
  k[2] = k0 ^ 0x6c7967656e657261ULL;
  k[3] = k1 ^ 0x7465646279746573ULL;
#endif

  // printf("hdrkey in setheader: ");
  // for(int i=0; i<32; i++){
  //   printf("%d,", hdrkey[i]);
  // }
  // printf("\n");


  setkeys(keys, hdrkey);
}

// edge endpoint in cuckoo graph with partition bit
edge_t sipnode_(siphash_keys *keys, edge_t edge, u32 uorv) {
  return sipnode(keys, edge, uorv) << 1 | uorv;
}
