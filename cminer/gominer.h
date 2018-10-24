#include <stdint.h>
#define result_t uint32_t
#ifdef __cplusplus
extern "C"
{
#endif
    uint8_t CuckooSolve(uint8_t *header, uint32_t header_len, uint64_t nonce, result_t *result, uint32_t *result_len, uint8_t *target, uint8_t *hash);
    uint8_t CuckooVerify(uint8_t *header, uint32_t header_len, uint64_t nonce, result_t *result, uint8_t* target, uint8_t* hash);
    int32_t CuckooFindSolutions(uint8_t *header, uint64_t nonce, result_t *result, uint32_t resultBuffSize, uint32_t* solLength, uint32_t *numSol);
    int32_t CuckooVerifyHeaderNonceAndSolutions(uint8_t *header, uint64_t nonce, result_t *result) ;
    int32_t CuckooVerifyProof(uint8_t *header, uint64_t nonce, result_t *result, uint8_t proofSize, uint8_t edgebits);
    void CuckooInit(uint32_t nthread, uint32_t nInstances);
    void CuckooFinalize();
#ifdef __cplusplus
}
#endif
