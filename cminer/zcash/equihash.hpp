
#pragma once

#define _GNU_SOURCE 1 /* memrchr */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>
#include <vector>

//opencl for different platforms

#ifdef _WIN32
#include <CL/cl.h>
#elif defined __unix__
#include <CL/cl.h>
#elif defined __APPLE__
#include <OpenCL/OpenCL.h>
#endif

typedef uint8_t uchar;
typedef uint32_t uint;
#include "param.h"

using namespace std;

class equiSolver{
private:
    int verbose = 0;
    uint32_t show_encoded = 0;
    uint64_t nr_nonces = 1;
    uint32_t do_list_devices = 0;
    uint32_t gpu_to_use = 0;
    uint32_t mining = 0;

    vector<uint8_t> header;
    //uint8_t* header;

    //=== openCL variables ===
    cl_platform_id plat_id = 0;
    cl_device_id dev_id = 0;
    cl_context context;
    cl_command_queue queue;

    cl_program program;
    const char *source;
    size_t source_len;

    //kernels
    cl_kernel k_rounds[PARAM_K];    //solution round
    cl_kernel k_init_ht;            //initialize
    cl_kernel k_sols;               //verify solution

public:
    void loadConfig();
    void getHeader(uint8_t* _header, int header_len);
    void getHeader(int header_len, const char* hex);
    void initOpenCL();
    void runOpenCL();
    void verifySol();
    void release();

    void solveEquihash();
    

};