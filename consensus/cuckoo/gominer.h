#ifdef __cplusplus
extern "C"
{
#endif
#define uint unsigned int
    void CuckooInit();
    unsigned char CuckooSolve(char *header, uint header_len, uint nonce, uint *result, uint *result_len, unsigned char* target,unsigned char* hash);
    unsigned char CuckooVerify(char *header, uint header_len, uint nonce, uint *result, unsigned char* target, unsigned char* hash);
    void CuckooRelease();

#ifdef __cplusplus
}
#endif