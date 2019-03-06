// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp
#include <cstdio>
#include "cuckoo.h"
namespace cuckoo {

const char *errstr[] = { "BAD", "OK", "wrong header length", "edge too big", "edges not ascending", "endpoints don't match up", "branch in cycle", "cycle dead ends", "cycle too short"};

node_t sipnode(siphash_keys *keys, edge_t edge, u32 uorv) {
  return siphash24(keys, 2*edge + uorv) & EDGEMASK;
}

int verify(edge_t edges[PROOFSIZE], siphash_keys *keys) {
  // printf("cuckoo.c edges: \n");
  // for (uint32_t i = 0; i < PROOFSIZE; i++) {
  //     printf(" %d", edges[i]);
  // }
  // printf("\n");
  // printf("sig: \n");
  // printf(" %d %d %d %d", keys->k0, keys->k1, keys->k2, keys->k3);
  // printf("\n");
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

edge_t sipnode_(siphash_keys *keys, edge_t edge, u32 uorv) {
  return sipnode(keys, edge, uorv) << 1 | uorv;
}

};

#ifdef STANDALONE_CUCKOO_TEST

#include <inttypes.h> // for SCNx64 macro
#include <stdio.h>    // printf/scanf
#include <stdlib.h>   // exit
#include <unistd.h>   // getopt
#include <assert.h>   // d'uh

// arbitrary length of header hashed into siphash key
#define HEADERLEN 40

int main(int argc, char **argv) {
  const char *header = "";
  int nonce = 0;
  int c;
  while ((c = getopt (argc, argv, "h:n:")) != -1) {
    switch (c) {
      case 'h':
        header = optarg;
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
    }
  }
  char headernonce[HEADERLEN];
  u32 hdrlen = strlen(header);
  memcpy(headernonce, header, hdrlen);
  memset(headernonce+hdrlen, 0, sizeof(headernonce)-hdrlen);
  // TODO fillin nonce
  siphash_keys keys;
  setheader(headernonce, sizeof(headernonce), &keys);
  printf("Verifying size %d proof for cuckoo%d(\"%s\",%d)\n",
               PROOFSIZE, EDGEBITS+1, header, nonce);
  for (int nsols=0; scanf(" Solution") == 0; nsols++) {
    edge_t nonces[PROOFSIZE];
    for (int n = 0; n < PROOFSIZE; n++) {
      u64 nonce;
      int nscan = scanf(" %" SCNx64, &nonce);
      assert(nscan == 1);
      nonces[n] = nonce;
    }
    int pow_rc = verify(nonces, &keys);
    if (pow_rc == POW_OK) {
      printf("Verified with cyclehash ");
      unsigned char cyclehash[32];
      blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)nonces, sizeof(nonces), 0, 0);
      for (int i=0; i<32; i++)
        printf("%02x", cyclehash[i]);
      printf("\n");
    } else {
      printf("FAILED due to %s\n", errstr[pow_rc]);
    }
  }
  return 0;
}

#endif
