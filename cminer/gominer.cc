#include <iostream>
#include <string.h>
#include <pthread.h>
#include "gominer.h"
#include "cuckoo/cuckoo.h"

int32_t CuckooVerifyHeaderNonceAndSolutions(uint8_t *header, uint64_t nonce, result_t *result) {
    uint64_t littleEndianNonce = htole64(nonce);
    char headerBuf[40];
    memcpy(headerBuf, header, 32);
    memcpy(headerBuf + 32, static_cast<uint64_t*>(&littleEndianNonce), sizeof(nonce));
    // for (uint32_t i = 0; i < header_len; i++)
    //     printf(" %d", headernonce[i]);
    // printf("\n");
    siphash_keys key;
    cuckoo::setheader(headerBuf, 40, &key);
    int res = cuckoo::verify(result, &key);
    return res;
}

