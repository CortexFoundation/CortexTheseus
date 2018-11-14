#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
}
#define CUDA_VERSION 9000
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>


#define SIGN(X) ((X)<0?-1:1)
#define ABS(X) ((X)<0?(-(X)):(X))
__global__ void convert_int32_int8(char* output, int* input, int size, char shift_bit){
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id>=size){
        return;
    }

    int r;
    r = input[id]/(1<<(shift_bit));
        // if (input[id]%2 == )
    // }
    // if (shift_bit>0 && (ABS(input[id])>>(shift_bit-1))%2==0 && input[id]<0){
    //     r+=1;git
    // }
    // if (shift_bit>0 && (ABS(input[id])>>(shift_bit-1))%2==1 && input[id]>0){
    //     r+=1;
    // }
    // if (id == 1)
    // printf("%d %d %d\n",input[id],shift_bit,r);
    // if (shift_bit == 0){
    // }else{
    //     if (ABS(input[id]>>(shift_bit-1))%2==1){
    //         r = SIGN(r)*(ABS(r>>shift_bit)+1);
    //     } else{
    //         r = r>>shift_bit;
    //     }
    // }
    r = r>127?127:r;
    r = r<-128?-128:r;
    output[id] = (char)(r);
    return;
}
extern "C" void cudaScale(char* output, int* input, int size, char shift_bit){
    if (shift_bit == -1)
    {
        thrust::device_ptr<int> d_vec = thrust::device_pointer_cast(input);
        int max = *(thrust::max_element(d_vec, d_vec + size));
        int min = *(thrust::min_element(d_vec, d_vec + size));
        if (ABS(min)>max)
            max = -min;
        char sb = 0;
        while (max>=128){
            max/=2;
            sb++;
        }
        // printf("%d %d\n",min,max);
        convert_int32_int8<<<cuda_gridsize(size), BLOCK>>>(output,input,size,sb);
    } else
        convert_int32_int8<<<cuda_gridsize(size), BLOCK>>>(output,input,size,shift_bit);

    check_error(cudaPeekAtLastError());
}