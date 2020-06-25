#include "cuckoo.h"
#include <stdio.h>
#include <assert.h>
namespace cuckoogpu {
node_t sipnode(siphash_keys *keys, edge_t edge, u32 uorv) {
  return siphash24(keys, 2*edge + uorv) & EDGEMASK;
}

const char *errstr[] = { "OK", "wrong header length", "edge too big", "edges not ascending", "endpoints don't match up", "branch in cycle", "cycle dead ends", "cycle too short"};

int verify(edge_t edges[PROOFSIZE], siphash_keys *keys) {
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

u64 sipblock(siphash_keys *key, const node_t edge, u64 *buf) {
	siphash_keys keys = *key;
  u64 v0 = keys.k0, v1 = keys.k1, v2 = keys.k2, v3 = keys.k3;

  edge_t edge0 = edge & ~EDGE_BLOCK_MASK;
  u32 i;
  for (i=0; i < EDGE_BLOCK_SIZE; i++) {
    //shs.hash24(edge0 + i);
	  edge_t nonce = edge0 + i;
	v3^=nonce;
	SIPROUND; SIPROUND;
	SIPROUND; SIPROUND;
	v0 ^= nonce;
	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;

//    buf[i] = shs.xor_lanes();
	buf[i] = (v0 ^ v1) ^ (v2  ^ v3);
  }
  const u64 last = buf[EDGE_BLOCK_MASK];
  for (u32 i=0; i < EDGE_BLOCK_MASK; i++)
    buf[i] ^= last;
  return buf[edge & EDGE_BLOCK_MASK];
}

int verify_proof_cuckaroo(edge_t* edges, siphash_keys *keys) {
  const uint32_t edgemask = ((edge_t)((node_t)1 << EDGEBITS) - 1);
  u64 sips[EDGE_BLOCK_SIZE];
  node_t uvs[2*PROOFSIZE];
  node_t xor0 = 0, xor1  =0;
  for (u32 n = 0; n < PROOFSIZE; n++) {
    if (edges[n] > edgemask)
      return POW_TOO_BIG;
    if (n && edges[n] <= edges[n-1])
      return POW_TOO_SMALL;
//    xor0 ^= uvs[2*n  ] = siphash24(keys, 2 * edges[n] + 0) & edgemask;
//    xor1 ^= uvs[2*n+1] = siphash24(keys, 2 * edges[n] + 1) & edgemask;
//
    u64 edge = sipblock(keys, edges[n], sips);
    xor0 ^= uvs[2*n  ] = edge & EDGEMASK;
    xor1 ^= uvs[2*n+1] = (edge >> 32) & EDGEMASK;
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

int verify_proof(edge_t* edges, siphash_keys *keys) {
  const uint32_t edgemask = ((edge_t)((node_t)1 << EDGEBITS) - 1);
  node_t uvs[2*PROOFSIZE];
  node_t xor0 = 0, xor1  =0;
  for (u32 n = 0; n < PROOFSIZE; n++) {
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
//  printf("call blake2b %d %d %d %d\n", hdrkey, sizeof(hdrkey), header, headerlen);
  blake2b((void *)hdrkey, sizeof(hdrkey), (const void *)header, headerlen, 0, 0);
//  assert(0);
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
  setkeys(keys, hdrkey);
}

// edge endpoint in cuckoo graph with partition bit
edge_t sipnode_(siphash_keys *keys, edge_t edge, u32 uorv) {
  return sipnode(keys, edge, uorv) << 1 | uorv;
}
};
/*void print_log(const char *fmt, ...) {
	if (SQUASH_OUTPUT) return;
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);

}*/
