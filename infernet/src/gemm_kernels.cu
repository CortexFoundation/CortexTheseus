#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void addPaddingA(char* h_A_new,char* h_A,int new_N,int new_M,int old_N,int old_M, int TA){
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id>=new_N*new_M){
        return;
    }
    h_A_new[id] = 0;
    int new_i = id / new_M,new_j = id %new_M;
    if (new_j>=old_M)
        return;
    if (TA==1)
    {
        if (id==1)
        h_A_new[id] = 1;
        if (id==0)
        h_A_new[id] = 2;
    }
    else
        h_A_new[id] = h_A[new_i*old_M+new_j];
    return;
}

__global__ void addPaddingB(char* h_B_new,char* h_B,int new_N,int new_M,int old_N, int old_M,int TB){
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id>=new_N*new_M){
        return;
    }
    h_B_new[id] = 0;
    if (TB == 0)
    {
        int new_i = id / new_M,new_j = id %new_M;
        if (new_i>=old_N || new_j>=old_M)
            return;
        h_B_new[id] = h_B[new_i*old_M+new_j];
    } else{
        int new_i = id / new_M,new_j = id %new_M;
       
        if (new_i>=old_N || new_j>=old_M)
            return;
    }
    return;
}
__global__ void rmPadding(int* h_C_old,int* h_C,int new_N,int new_M,int old_N,int old_M){
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id>=old_N*old_M){
        return;
    }
    h_C_old[id] = 0;
    int old_i = id / old_M,old_j = id %old_M;
    h_C_old[id] = h_C[old_i*new_M+old_j];
    return;
}
__global__ void addBias(int* h_C, int size,char* bias){
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id>=size){
        return;
    }
    h_C[id] += bias[id];
    return;
}

void addPaddingA_gpu(char* h_A_new,char* h_A,int new_N,int new_M,int old_N,int old_M,int TA){

    addPaddingA<<<cuda_gridsize(new_M*new_N),BLOCK>>>(h_A_new,h_A,new_N,new_M,old_N,old_M,TA);
    check_error(cudaPeekAtLastError());
    return ;
}

void addPaddingB_gpu(char* h_B_new,char* h_B,int new_N,int new_M,int old_N,int old_M,int TB){


    addPaddingB<<<cuda_gridsize(new_M*new_N),BLOCK>>>(h_B_new,h_B,new_N,new_M,old_N,old_M,TB);
    check_error(cudaPeekAtLastError());
    return ;
}
void rmPadding_gpu(int* h_C,int* h_C_new,int new_N,int new_M,int old_N,int old_M){

    rmPadding<<<cuda_gridsize(old_N*old_M),BLOCK>>>(h_C,h_C_new,new_N,new_M,old_N,old_M);
    check_error(cudaPeekAtLastError());
    return ;
}
void add_bias_gpu_fc(int* h_C,int size,char* bias){

    addBias<<<cuda_gridsize(size),BLOCK>>>(h_C,size,bias);
    check_error(cudaPeekAtLastError());
    return ;
}

