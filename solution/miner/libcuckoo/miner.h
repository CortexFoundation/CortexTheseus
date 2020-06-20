#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif
int32_t FindSolutionsByGPU(
        uint8_t *header,
        uint64_t nonce,
	uint32_t threadId);
int32_t FindCycles(
	uint32_t threadId,
	uint32_t nedges,
	uint32_t *result,
	uint32_t resultBufferSize,
	uint32_t *solLength,
	uint32_t *numSol);
void CuckooInitialize(uint32_t *devices, uint32_t deviceNum, int selected, int printDeviceInfo);
void CuckooInitializeCPU(uint32_t *devices, uint32_t deviceNum, int selected, int printDeviceInfo);
void CuckooFinalize();
void CuckooFinalizeCPU();
int monitor(unsigned int device_count, unsigned int *fanSpeeds, unsigned int *temperatures);
int32_t RunSolverOnCPU(
        uint8_t *header,
        uint64_t nonce,
        uint32_t *result,
        uint32_t resultBuffSize,
        uint32_t *solLength,
        uint32_t *numSol);

#define result_t uint32_t
int32_t CuckooVerifyProof(uint8_t *header, uint64_t nonce, result_t *result);
int32_t CuckooVerifyProof_cuckaroo(uint8_t *header, uint64_t nonce, result_t *result);

#ifdef __cplusplus
}
#endif
