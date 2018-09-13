#include <iostream>
#include <string.h>
#include <pthread.h>
#include "gominer.h"
#include "cuckoo/cuckoo.h"

int32_t CuckooVerifyHeaderNonceAndSolutions(uint8_t *header, uint32_t header_len, uint32_t nonce, result_t *result) {
#ifndef HEADERLEN
#define HEADERLEN 80
#define HEADERLEN_TEMP_DEFINED
#endif
    char headernonce[HEADERLEN];
    memcpy(headernonce, header, header_len);
    ((u32 *)headernonce)[header_len/sizeof(u32)-1] = htole32(nonce);
    // for (uint32_t i = 0; i < header_len; i++)
    //     printf(" %d", headernonce[i]);
    // printf("\n");
    siphash_keys key;
    cuckoo::setheader(headernonce, header_len, &key);

    int res = cuckoo::verify(result, &key);
    return res;
#ifdef HEADERLEN_TEMP_DEFINED
#undef HEADERLEN_TEMP_DEFINED
#undef HEADERLEN
#endif
}

