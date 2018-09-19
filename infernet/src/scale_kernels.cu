#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
}
__global__ void convert_int32_int8(char* output, int* input, int size, char shift_bit){
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id>=size){
        return;
    }
    int r = input[id] >> shift_bit;
    r = r>127?127:r;
    r = r<-128?-128:r;
    output[id] = (char)(r);
    return;
}
extern "C" void cudaScale(char* output, int* input, int size, char shift_bit){
    convert_int32_int8<<<cuda_gridsize(size), BLOCK>>>(output,input,size,shift_bit);

    check_error(cudaPeekAtLastError());
}