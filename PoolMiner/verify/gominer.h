#ifndef GOMINER_H
#define GOMINER_H

#include <stdint.h>
#define result_t uint32_t

int32_t CuckooVerifyProof(uint8_t *header, uint64_t nonce, result_t *result, uint8_t proofSize, uint8_t edgebits);

int32_t CuckooVerifyProof_cuckaroo(uint8_t *header, uint64_t nonce, result_t *result, uint8_t proofSize, uint8_t edgebits);
#endif // GOMINER_H
