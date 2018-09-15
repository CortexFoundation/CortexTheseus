#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif
int32_t CuckooFindSolutionsCuda(
        uint8_t *header,
        uint64_t nonce,
        uint32_t *result,
        uint32_t resultBuffSize,
        uint32_t *solLength,
        uint32_t *numSol);
void CuckooInitialize();
void CuckooFinalize();
#ifdef __cplusplus
}
#endif
