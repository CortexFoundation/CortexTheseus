#include "minerBot.h"
#include "gominer.h"
#include <thread>
#include <vector>

using std::thread;

void run();

int main(int argc, char** argv){

    printf("Welcome to Cortex Mining.\n");

    CuckooInit(1);
    int num = 20;
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
    char a[100];
    char b[100];
    for(uint32_t i = 0; i < 100; i++){
        a[i] = 10;
        b[i] = 20;
    }
    uint32_t result[200];
    uchar t[32];
    uchar h[32];
    for(uint32_t i = 0; i < 32; i++){
        t[i] = 255;
    }
    uint32_t result_l = 4;
    printf("let's find the bug!\n");

    uint32_t nonce = 0;
    int debugsz = 80;
    while(true){
        printf("solving nonce %u...\n", nonce);
        uchar r = CuckooSolve(
                a,
                debugsz,
                nonce,
                result,
                &result_l,
                t,
                h);
        if(r){
            printf("with a result we verify...\n");
            uchar v = CuckooVerify(
                    a, debugsz, nonce, result, t, h
                    );
            if(v)
                break;
        }
        nonce ++;
    }

    printf("over.\n");
}
