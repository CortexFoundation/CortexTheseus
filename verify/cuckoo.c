// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp
#include <stdio.h>
#include "cuckoo.h"

const char *errstr[] = {"BAD", "OK", "wrong header length", "edge too big", "edges not ascending", "endpoints don't match up", "branch in cycle", "cycle dead ends", "cycle too short"};

int verify_proof(edge_t* edges, uint8_t proof_size, uint8_t edgebits, siphash_keys *keys) {
  const uint32_t edgemask = ((edge_t)((node_t)1 << edgebits) - 1);
  node_t uvs[2*proof_size];
  node_t xor0 = 0, xor1  =0;
  for (u32 n = 0; n < proof_size; n++) {
    if (edges[n] > edgemask)
      return POW_TOO_BIG;
    if (n && edges[n] <= edges[n-1])
      return POW_TOO_SMALL;
    xor0 ^= uvs[2*n  ] = siphash24(keys, 2 * edges[n] + 0) & edgemask;
    xor1 ^= uvs[2*n+1] = siphash24(keys, 2 * edges[n] + 1) & edgemask;
  }
  if (xor0|xor1)              // optional check for obviously bad proofs
    return POW_NON_MATCHING;
  u32 n = 0, i = 0, j;
  do {                        // follow cycle
    for (u32 k = j = i; (k = (k+2) % (2*proof_size)) != i; ) {
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
  return n == proof_size ? POW_OK : POW_SHORT_CYCLE;
}

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
