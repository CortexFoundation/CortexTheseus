#ifndef XCORTEX_VERIFY_H
#define XCORTEX_VERIFY_H

#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif

int Verify(const uint64_t nonce, const uint8_t* header, const uint8_t*difficulty);
int Verify2(const uint64_t nonce, const uint8_t* header, const uint8_t *difficulty, const uint8_t *shareTarget, const uint8_t *blockTarget, int32_t*ret);

#ifdef __cplusplus
}
#endif

#endif
