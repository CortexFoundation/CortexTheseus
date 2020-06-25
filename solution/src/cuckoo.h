// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp
#ifndef CUCKOO_H
#define CUCKOO_H

#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset
#include "blake2.h"
#include "siphash.h"

#include <stdarg.h>
#include <chrono>
#include <ctime>
#include "blake2.h"
#include "siphash.h"
#include <stdio.h>

typedef uint32_t u32;
typedef uint64_t u64;

#ifndef MAX_SOLS
#define MAX_SOLS 4
#endif

#ifndef EDGE_BLOCK_BITS
#define EDGE_BLOCK_BITS 6
#endif
#define EDGE_BLOCK_SIZE (1 << EDGE_BLOCK_BITS)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)

// proof-of-work parameters
#ifndef EDGEBITS
// the main parameter is the 2-log of the graph size,
// which is the size in bits of the node identifiers
#define EDGEBITS 30
#endif
#ifndef PROOFSIZE
// the next most important parameter is the (even) length
// of the cycle to be found. a minimum of 12 is recommended
#define PROOFSIZE 42
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

#ifndef C_CALL_CONVENTION
#define C_CALL_CONVENTION 0
#endif

// convention to prepend to called functions
#if C_CALL_CONVENTION
#define CALL_CONVENTION extern "C"
#else
#define CALL_CONVENTION
#endif

// Ability to squash printf output at compile time, if desired
#ifndef SQUASH_OUTPUT
#define SQUASH_OUTPUT 0
#endif

namespace cuckoogpu {
  // generate edge endpoint in cuckoo graph without partition bit
  node_t sipnode(siphash_keys *keys, edge_t edge, u32 uorv);

  enum verify_code { POW_BAD, POW_OK, POW_HEADER_LENGTH, POW_TOO_BIG, POW_TOO_SMALL, POW_NON_MATCHING, POW_BRANCH, POW_DEAD_END, POW_SHORT_CYCLE};
  extern const char *errstr[];

  // verify that edges are ascending and form a cycle in header-generated graph
  int verify(edge_t edges[PROOFSIZE], siphash_keys *keys);
  int verify_proof(edge_t* edges, siphash_keys *keys);

  int verify_proof_cuckaroo(edge_t* edges, siphash_keys *keys);

  // convenience function for extracting siphash keys from header
  void setheader(const char *header, const u32 headerlen, siphash_keys *keys);

  // edge endpoint in cuckoo graph with partition bit
  edge_t sipnode_(siphash_keys *keys, edge_t edge, u32 uorv);
}


/*void print_log(const char *fmt, ...);
void print_log(const char *fmt, ...) {
	if (SQUASH_OUTPUT) return;
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);

}
*/
#endif
