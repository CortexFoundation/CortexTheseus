#include "CuckooSolver.h"

int main(){

    CuckooSolver test;
    test.setNthreads(2);
    test.initSolver();
    test.setHeaderNonce(NULL, 0, 63);
    test.solve();
    test.release();

}
