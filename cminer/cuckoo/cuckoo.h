// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp
#ifndef H_CUCKOO
#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset

#ifdef SIPHASH_COMPAT
#include <stdio.h>
#endif

#include "param.h"
#include "blake2.h"
#include "siphash.h"

// proof-of-work parameters
#ifndef EDGEBITS
// the main parameter is the 2-log of the graph size,
// which is the size in bits of the node identifiers
#define EDGEBITS 29
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
node_t sipnode(siphash_keys *keys, edge_t edge, u32 uorv);

enum verify_code {
    POW_BAD = 0,
    POW_OK = 1,
    POW_HEADER_LENGTH = 2,
    POW_TOO_BIG = 3,
    POW_TOO_SMALL = 4,
    POW_NON_MATCHING = 5,
    POW_BRANCH = 6,
    POW_DEAD_END = 7,
    POW_SHORT_CYCLE = 8
};
extern const char *errstr[];

// verify that edges are ascending and form a cycle in header-generated graph
int verify(edge_t edges[PROOFSIZE], siphash_keys *keys);

// convenience function for extracting siphash keys from header
void setheader(const char *header, const u32 headerlen, siphash_keys *keys);

// edge endpoint in cuckoo graph with partition bit
edge_t sipnode_(siphash_keys *keys, edge_t edge, u32 uorv);
#endif
