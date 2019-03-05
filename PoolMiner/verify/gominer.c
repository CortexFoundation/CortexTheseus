#include <string.h>
#include <pthread.h>
#include "gominer.h"
#include "cuckoo.h"

int32_t CuckooVerifyProof(uint8_t *header, uint64_t nonce, result_t *result, uint8_t proofSize, uint8_t edgebits) {
    uint64_t littleEndianNonce = htole64(nonce);
    char headerBuf[40];
    memcpy(headerBuf, header, 32);
    memcpy(headerBuf + 32, (uint64_t*)(&littleEndianNonce), sizeof(nonce));
    siphash_keys key;
    setheader(headerBuf, 40, &key);
    int res = verify_proof(result, proofSize, edgebits, &key);
    return res;
}

int32_t CuckooVerifyProof_cuckaroo(uint8_t *header, uint64_t nonce, result_t *result, uint8_t proofSize, uint8_t edgebits) {
    uint64_t littleEndianNonce = htole64(nonce);
    char headerBuf[40];
    memcpy(headerBuf, header, 32);
    memcpy(headerBuf + 32, (uint64_t*)(&littleEndianNonce), sizeof(nonce));
    siphash_keys key;
    setheader(headerBuf, 40, &key);
    int res = verify_proof_cuckaroo(result, proofSize, edgebits, &key);
    return res;
}
