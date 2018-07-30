#ifdef __cplusplus
extern "C"
{
#endif
#define uint unsigned int
    void CuckooInit();
    void CuckooSolve(char *header, uint header_len, uint nonce, uint *result, uint *result_len);
    unsigned char CuckooVerify(char *header, uint header_len, uint nonce, uint *result);
    void CuckooRelease();

#ifdef __cplusplus
}
#endif