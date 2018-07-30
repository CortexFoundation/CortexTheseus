#include "cuckooSolver.h"

int main(){

    cuckooSolver test;
    test.initSolver();
    test.setHeaderNonce(NULL, 0, 63);
    test.solve();
    test.release();

}