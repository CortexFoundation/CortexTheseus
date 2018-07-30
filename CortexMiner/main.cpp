
#include "minerBot.h"
//#include "./zcash/equihash.hpp"

int main(int argc, char** argv){

    printf("Welcome to Cortex Mining.\n");

    minerBot bot;
    //bot.loadParam();
    //bot.start();
    
    //bot.testEquihash();
    bot.testCuckoo();

    return 0;
}