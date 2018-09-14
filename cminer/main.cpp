#include <thread>
#include <vector>
#include "gominer.h"

#ifndef VERIFY_ONLY

#include "minerBot.h"

using std::thread;

void run();

int main(int argc, char** argv){

    printf("Welcome to Cortex Mining.\n");

    CuckooInit(1, 5);
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
    // uchar v = CuckooVerify( a + 0, debugsz, nonce, result, t, h);
    // printf("%d\n", v);
    // return;
    uint8_t a[80] = {52,105,200,55,246,196,13,239,1,106,158,159,167,174,165,127,187,126,234,16,191,241,71,244,221,159,19,104,170,183,82,216};
    uint32_t result[42] =  {3470748,9942985,11832470,14637222,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    uint8_t t[32] = {255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    uint8_t h[32] = {4,42,239,229,57,70,81,116,187,90,204,230,82,230,150,104,56,77,231,109,218,198,22,6,192,78,177,63,202,132,34,245};
    uint32_t nonce = 3;

    while(true){
        uint8_t tmp[80];
        for (uint32_t i = 0; i < sizeof(a); i++) {
            tmp[i] = a[i];
        }

        printf("solving nonce %u...\n", nonce);
        uchar r = CuckooSolve(
                tmp + 0,
                debugsz,
                nonce,
                result,
                &result_l,
                t,
                h);
        printf("nonce: %d\nResult: ", nonce);
        for (uint32_t i = 0; i < 42; i++) {
            printf(" %d", result[i]);
        }
        printf("\nTarget: ");
        for (uint32_t i = 0; i < sizeof(t); i++) {
            printf(" %d", t[i]);
        }
        printf("\n");
        if(r){
            for (uint32_t i = 0; i < sizeof(a); i++) {
                tmp[i] = a[i];
            }
            printf("with a result we verify...\n");
            uchar v = CuckooVerify(
                    tmp + 0,
                    debugsz, nonce, result, t, h
                    );
                break;
        }
        nonce ++;
    }

    printf("over.\n");
}
#else

#include <cstdio>
#include "cuckoo/cuckoo.h"

int main() {
    printf("res: %d\n", CuckooVerifyHeaderNonceAndSolutions(
        a + 0,
        32,
        nonce,
        &result[0]));
    return 0;
}

#endif
