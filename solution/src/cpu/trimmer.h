

#ifndef XBITS
// 7 seems to give best performance
#define XBITS 7
#endif

#define YBITS XBITS

// size in bytes of a big bucket entry
#ifndef BIGSIZE
#if EDGEBITS <= 15
#define BIGSIZE 4
// no compression needed
#define COMPRESSROUND 0
#else
#define BIGSIZE 5
// YZ compression round; must be even
#ifndef COMPRESSROUND
#define COMPRESSROUND 14
#endif
#endif
#endif
// size in bytes of a small bucket entry
#define SMALLSIZE BIGSIZE

// initial entries could be smaller at percent or two slowdown
#ifndef BIGSIZE0
#if EDGEBITS < 30 && !defined SAVEEDGES
#define BIGSIZE0 4
#else
#define BIGSIZE0 BIGSIZE
#endif
#endif
// but they may need syncing entries
#if BIGSIZE0 == 4 && EDGEBITS > 27
#define NEEDSYNC
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#if EDGEBITS >= 30
typedef u64 offset_t;
#else
typedef u32 offset_t;
#endif

#if BIGSIZE0 > 4
typedef u64 BIGTYPE0;
#else
typedef u32 BIGTYPE0;
#endif

// node bits have two groups of bucketbits (X for big and Y for small) and a remaining group Z of degree bits
const u32 NX        = 1 << XBITS;
const u32 XMASK     = NX - 1;
const u32 NY        = 1 << YBITS;
const u32 YMASK     = NY - 1;
const u32 XYBITS    = XBITS + YBITS;
const u32 NXY       = 1 << XYBITS;
const u32 ZBITS     = EDGEBITS - XYBITS;
const u32 NZ        = 1 << ZBITS;
const u32 ZMASK     = NZ - 1;
const u32 YZBITS    = EDGEBITS - XBITS;
const u32 NYZ       = 1 << YZBITS;
const u32 YZMASK    = NYZ - 1;
const u32 YZ1BITS   = YZBITS < 15 ? YZBITS : 15;  // compressed YZ bits
const u32 NYZ1      = 1 << YZ1BITS;
const u32 YZ1MASK   = NYZ1 - 1;
const u32 Z1BITS    = YZ1BITS - YBITS;
const u32 NZ1       = 1 << Z1BITS;
const u32 Z1MASK    = NZ1 - 1;
const u32 YZ2BITS   = YZBITS < 11 ? YZBITS : 11;  // more compressed YZ bits
const u32 NYZ2      = 1 << YZ2BITS;
const u32 YZ2MASK   = NYZ2 - 1;
const u32 Z2BITS    = YZ2BITS - YBITS;
const u32 NZ2       = 1 << Z2BITS;
const u32 Z2MASK    = NZ2 - 1;
const u32 YZZBITS   = YZBITS + ZBITS;
const u32 YZZ1BITS  = YZ1BITS + ZBITS;

const u32 BIGSLOTBITS   = BIGSIZE * 8;
const u32 SMALLSLOTBITS = SMALLSIZE * 8;
const u64 BIGSLOTMASK   = (1ULL << BIGSLOTBITS) - 1ULL;
const u64 SMALLSLOTMASK = (1ULL << SMALLSLOTBITS) - 1ULL;
const u32 BIGSLOTBITS0  = BIGSIZE0 * 8;
const u64 BIGSLOTMASK0  = (1ULL << BIGSLOTBITS0) - 1ULL;
const u32 NONYZBITS     = BIGSLOTBITS0 - YZBITS;
const u32 NNONYZ        = 1 << NONYZBITS;
