// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp
#ifndef H_CUCKOO
#define H_CUCKOO

#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset

#include "blake2.h"
#include "siphash.h"

// proof-of-work parameters
#ifndef EDGEBITS
#define EDGEBITS 28
#endif

#ifndef PROOFSIZE
#define PROOFSIZE 12
#endif

typedef u32 edge_t;
typedef u32 node_t;

// #if EDGEBITS > 32
// typedef u64 edge_t;
// #else
// typedef u32 edge_t;
// #endif
// #if EDGEBITS > 31
// typedef u64 node_t;
// #else
// typedef u32 node_t;
// #endif

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

int verify_proof(edge_t* edges, uint8_t proof_size, uint8_t edgebits, siphash_keys *keys);

void setheader(const char *header, const u32 headerlen, siphash_keys *keys);
#endif
