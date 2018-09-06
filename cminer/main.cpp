#include <thread>
#include <vector>
#include "gominer.h"

uint8_t a[80] = {133,201,250,241,66,91,106,72,148,190,158,97,59,104,148,0,187,47,43,18,86,254,223,224,63,148,92,131,106,35,195,141};
uint32_t nonce =   1790697960 ;
uint32_t result[42] =  {239454,771580,10063629,13861271,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
uint8_t t[32] = {15,254,0,63,248,0,255,224,3,255,128,15,254,0,63,248,0,255,224,3,255,128,15,254,0,63,248,0,255,224};
uint8_t h[32] = {10,139,10,212,241,104,85,8,233,127,140,66,16,151,248,51,79,199,25,57,239,254,104,181,23,178,75,121,15,90,215,135};

#ifndef VERIFY_ONLY

#include "minerBot.h"

using std::thread;

void run();

int main(int argc, char** argv){

    printf("Welcome to Cortex Mining.\n");

    CuckooInit(1);
    int num = 1;
    vector<thread> threads;
    for (int i = 0; i < num; ++i) {
        threads.push_back(thread(run));
    }
    for (auto& thread: threads) {
        thread.join();
    }
    CuckooFinalize();
}
void run() {
    // uint32_t expected_result[] = {11318189 12740372 13520514 14228845}
    uint32_t result_l = 4;
    printf("let's find the bug!\n");

    int debugsz = 32;
    uchar v = CuckooVerify( a + 0, debugsz, nonce, result, t, h);
    printf("%d\n", v);
    return;

    while(true){
        printf("solving nonce %u...\n", nonce);
        uchar r = CuckooSolve(
                a + 0,
                debugsz,
                nonce,
                result,
                &result_l,
                t,
                h);
        printf("nounce: %d\n", nonce);
        for (uint32_t i = 0; i < sizeof(result); i++) {
            printf(" %d", result[i]);
        }
        printf("\n");
        for (uint32_t i = 0; i < sizeof(t); i++) {
            printf(" %d", t[i]);
        }
        printf("\n");
        if(r){
            printf("with a result we verify...\n");
            uchar v = CuckooVerify(
                    a + 0,
                    debugsz, nonce, result, t, h
                    );
            if(v)
                break;
        }
        nonce ++;
    }

    printf("over.\n");
}
#else

#include <cstdio>
#include "cuckoo/cuckoo.h"
int  CuckooVerifyHeaderNonceAndSolutions(char *header, uint32_t header_len, uint32_t nonce, result_t* result)
{
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
    setheader(headernonce, header_len, &key);

    int res = verify(result, &key);
    return res;
#ifdef HEADERLEN_TEMP_DEFINED
#undef HEADERLEN_TEMP_DEFINED
#undef HEADERLEN
#endif
}

int CuckooVerifySolutions(char *header, uint32_t header_len, result_t* result)
{
    siphash_keys key;
    setheader(header, header_len, &key);
    int res = verify(result, &key);
    return res;
}
int main() {
    printf("res: %d\n", CuckooVerifyHeaderNonceAndSolutions(
        reinterpret_cast<char*>(&a[0]),
        32,
        nonce,
        &result[0]));
    return 0;
}

#endif
