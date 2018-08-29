#include "param_define.h"

int main(){

    g_bcfg.setEdgeBits(27);
    g_bcfg.setProofsize(42);
    g_bcfg.updateParam();

    zbucket_d tb;
    tb.setBucketSize(g_bcfg.zbucketsize);
}