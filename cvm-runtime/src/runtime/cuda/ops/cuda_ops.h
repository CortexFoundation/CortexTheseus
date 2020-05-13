#ifndef CUDA_OP_H
#define CUDA_OP_H

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <memory>
#include <string.h>
#include <iostream>
#include <string>
#define DEBUG

namespace cvm{
namespace runtime{

const int NON_ERROR = 0;

const int ERROR_DIV_0 = 10;
const int ERROR_LOG_0 = 11;

const int ERROR_MALLOC = 20;
const int ERROR_MEMCPY = 21;
const int ERROR_MEMSET = 22;
const int ERROR_GET_PROPERTIES = 23;
const int ERROR_KERNEL = 24;
const int ERROR_PARAMS = 25;

//static cudaEvent_t start, stop;
//inline void start_time(){
//#ifdef CVM_PROFILING
//  cudaEventCreate(&start);
//  cudaEventCreate(&stop);
//  cudaEventRecord(start, 0);
//#endif
//}
//inline float get_used_time(){
//  float cost_time = 0;
//#ifdef CVM_PROFILING
//  cudaEventRecord(stop, 0);
//  cudaEventSynchronize(stop);
//  cudaEventElapsedTime(&cost_time, start, stop);
//#endif
//  return cost_time/1000.0;
//}

inline const char* check_cuda_error(cudaError_t error){
  if(error == cudaSuccess) return NULL;
  else return cudaGetErrorString(error);
}

//#define CVM_PRINT_CUDA_RESULT

const std::string DIR = "/tmp/zkh/ssd/gpu/";
template<typename T>
inline void print_to_file(const T *y, int32_t n, std::string filename){
#ifdef CVM_PRINT_CUDA_RESULT
  T *y_data = new T[n];
  cudaMemcpy(y_data, y, sizeof(T)*n, cudaMemcpyDeviceToHost);

  FILE *fp = fopen((DIR+filename).c_str(), "a+");

  int32_t min = y_data[0], max= y_data[0];
  for(uint64_t i = 0; i < n; i++){
    min = min > (int)y_data[i] ? (int)y_data[i] : min;
    max = max < (int)y_data[i] ? (int)y_data[i] : max;
  }
  //fprintf(fp, "%d %d\n", min, max);
  for(uint64_t i = 0; i < n; i++){
    fprintf(fp, "%d ", (int)y_data[i]);
  }
  fprintf(fp, "\n");
  fclose(fp);
  delete y_data;
#endif
}

#define MEMORY_LIMIT (512*1024*1024)

inline int32_t getGridSize(const int64_t n, const int32_t blockSize){
  int64_t tg = (n + blockSize - 1) / blockSize;
  return tg > 4096 ? 4096 : tg;
}

inline int32_t getShareMemorySize(const int32_t device_id, int&error_code){
  static int32_t sharedMemPerBlock = 0;
  if(sharedMemPerBlock == 0){
    cudaDeviceProp prop;
    cudaError_t status = cudaGetDeviceProperties(&prop, device_id);
    if(status != cudaSuccess){
        error_code = ERROR_GET_PROPERTIES;
        return -1;
    }
    sharedMemPerBlock = prop.sharedMemPerBlock;
  }
  return sharedMemPerBlock;
}
inline int32_t getFreeMemorySize(const int32_t device_id, int&error_code){
  size_t freeSize = 0, totalSize = 0;
  cudaError_t status = cudaMemGetInfo(&freeSize, &totalSize);
  if(status != cudaSuccess){
    error_code = ERROR_GET_PROPERTIES;
    return -1;
  }
  return freeSize;
}

const char* cuda_elemwise_add(int32_t *a, int32_t *b, int32_t *c, uint64_t n, int& error_code);
const char* cuda_elemwise_sub(int32_t *a, int32_t *b, int32_t *c, uint64_t n, int& error_code);
const char* cuda_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t stride_w,
        int32_t dilation_h, int32_t dilation_w,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, 
        int32_t *ext_space,
        int32_t ext_space_size,
        int& error_code);
const char* cuda_depthwise_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t stride_w,
        int32_t dilation_h, int32_t dilation_w,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, int& error_code);
const char* cuda_groupwise_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t stride_w,
        int32_t dilation_h, int32_t dilation_w,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, int& error_code);
const char* cuda_max_pool(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t f_h, int32_t f_w,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t w,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, int& error_code);
const char* cuda_dense(
        int32_t *a,
        int32_t *b,
        int32_t *c,
        const int m, const int k, const int n, int32_t *bias, int& error_code);
const char* cuda_clip(const int32_t *x, int32_t *y, const uint64_t n, const int32_t max, const int32_t min, int& error_code);
const char* cuda_relu(const int32_t *x, int32_t *y, const uint64_t n, int& error_code);
const char* cuda_flatten(const int32_t *x, int32_t *y, const uint64_t n, int& error_code);
const char* cuda_broadcast_add(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, int& error_code);
const char* cuda_broadcast_sub(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, int& error_code);
const char* cuda_broadcast_mul(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, int& error_code);
const char* cuda_broadcast_div(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, int& error_code);
const char* cuda_broadcast_right_shift(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, int& error_code);
const char* cuda_broadcast_left_shift(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, int& error_code);
const char* cuda_broadcast_max(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, int& error_code);
const char* cuda_broadcast_greater(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, int& error_code);
const char* cuda_sum(const int32_t *x, int32_t *y, const uint64_t xsize, const uint64_t ysize,
    const int64_t *xshape, const int64_t *yshape, const int32_t* realAxis, const int32_t* flag,
    const uint64_t *every_xdim_size, const int64_t axis_size,
    const int32_t xndim, const int32_t yndim, const int32_t axis_ndim, int& error_code);
const char* cuda_reshape(const int32_t *x, int32_t *y, uint64_t size, int& error_code);
const char* cuda_log(const int32_t *x, int32_t *y, const uint64_t n, int& error_code);
const char* cuda_abs(const int32_t *x, int32_t *y, const uint64_t n, int& error_code);
// const char* cuda_sqrt(const int32_t *x, int32_t *y, const uint64_t n, int& error_code);
const char* cuda_cvm_clip(const int32_t* x, const int32_t precision, int32_t *y, const uint64_t n, int& error_code);
const char* cuda_max(const int32_t *x, int32_t *y, const uint64_t xsize, const uint64_t ysize,
    const int64_t *xshape, const int64_t *yshape, const int32_t* realAxis, const int32_t* flag,
    const uint64_t *every_xdim_size, const int64_t axis_size,
    const int32_t xndim, const int32_t yndim, const int32_t axis_ndim, int& error_code);
const char* cuda_cvm_right_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const uint64_t n, int& error_code);
const char* cuda_cvm_left_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const uint64_t n, int& error_code);
const char* cuda_concatenate(int32_t **inputs, int64_t *ishapes, const int32_t ninput, const int32_t ndim, int32_t *output,const int64_t* oshape, const int32_t axis, int32_t* axisSize, int32_t *ext_space, int& error_code);
const char* cuda_bias_add(const int32_t *x_data, const int32_t * bias_data, int32_t *y_data,
        int64_t ysize, const int64_t *yshape, const int32_t ndim, const int32_t axis, int& error_code);
const char* cuda_repeat(const int32_t *x_data, int32_t *y_data, const int64_t *xshape,
        const int64_t *yshape, const uint64_t ysize, const int32_t xndim, const int32_t yndim, const int32_t axis, const int32_t repeat, int& error_code);
const char* cuda_upsampling_nearest(const int32_t *x_data, int32_t *y_data, const uint32_t scale, const int32_t ih, const int32_t iw,
    const uint32_t oh, const uint32_t ow, const uint32_t batch, const uint32_t channel, int& error_code);
const char* cuda_negative(const int32_t *x_data, int32_t *y_data, uint64_t n, int& error_code);
const char* cuda_tile(const int32_t *x_data, int32_t *y_data, const uint64_t ysize, const int32_t yndim, const int32_t xndim,
        const int64_t *xshape, const int64_t *yshape, int& error_code);
// const char* cuda_pad(const int32_t *x_data, const int32_t *pad_data, const int32_t pad_value, int32_t *y_data,
    // const int64_t *xshape, const int64_t *yshape, const int32_t xndim, const uint64_t ysize, int& error_code);
const char *cuda_expand_dims(const int32_t *ishape_data, int32_t *oshape_data, const int32_t axis, const uint64_t n, int& error_code);
const char *cuda_squeeze(const int32_t *ishape_data, int32_t *oshape_data, const uint64_t n, int& error_code);
const char* cuda_transpose(const int32_t *x_data, const int64_t *axes_data, int32_t *y_data,
        const int64_t *xshape, const int64_t *yshape, const int32_t ndim, const uint64_t ysize, const int32_t axes_ndim, int& error_code);
const char* cuda_stride_slice(const int32_t *x_data, int32_t *y_data, const int64_t *begin_data,
        const int32_t begin_ndim, const int64_t *step_data, const int64_t *xshape, const int64_t *yshape,
        const int32_t step_ndim, const int32_t y_ndim, const uint64_t ysize, const int32_t x_ndim, int& error_code);
const char* cuda_slice_like(const int32_t *x_data, int32_t *y_data, const int64_t *xshape, const int64_t *yshape,
        const uint64_t ysize, const int32_t ndim, int& error_code);
const char* cuda_get_valid_counts(int32_t *x_data, int32_t *y_data, int32_t *valid_count_data,
        const int32_t n, const int32_t k,
        const int32_t score_threshold, const int32_t batchs, int32_t *ext_space, int& error_code);
const char *cuda_non_max_suppression(int32_t *d_x_data, const int32_t *d_valid_count_data, int32_t *d_y_data, const int32_t batchs, const int32_t n, const int32_t k,
        const int32_t max_output_size, const int32_t iou_threshold, const int32_t topk,
        const int32_t coord_start, const int32_t score_index, const int32_t id_index, const bool force_suppress, int32_t *ext_space, int& error_code);
const char* cuda_take(const int32_t *x_data, const int32_t *indices_data, int32_t *y_data,
        const int64_t *xshape, const int64_t *yshape, const int64_t *indices_shape, const int32_t yndim,
        const int32_t xndim, const int32_t indices_ndim, const uint64_t ysize, const int32_t axis, int& error_code);
const char* cuda_take(const int32_t *x_data, const int32_t *indices_data, int32_t *y_data, const uint64_t ysize, const uint64_t xsize, int& error_code);
const char* cuda_where(const int32_t *x_data, const int32_t *y_data, const int32_t *condition_data, int32_t *result_data, bool same_shape, uint64_t n, uint64_t shape0, int& error_code);



inline void cvm_cuda_malloc(void **p, size_t size){
  cudaError_t status = cudaMalloc(p, size);
  if(status != cudaSuccess){
    throw ERROR_MALLOC;
  }
}
inline void cvm_cuda_memcpy(void *dst, void* src, size_t size, cudaMemcpyKind flag){
  cudaError_t status = cudaMemcpy(dst, src, size, flag);
  if(status != cudaSuccess){
    throw ERROR_MEMCPY;
  }
}

#define MAX_DIM 6
template<typename T>
inline void get_cuda_shape(const T *ishape, const int dim, T *oshape){
  int shift = MAX_DIM - dim;
  for(int i = 0; i < MAX_DIM; i++){
    oshape[i] = 1;
    if(i >= shift){
      oshape[i] = ishape[i - shift];
    }
  }
}

inline int32_t get_multi8to64_size(int32_t n){
  return (n + 7) / 8 * 8;
}
inline int32_t get_multi32to64_size(int32_t n){
  return (n + 1) / 2 * 2;
}
inline int32_t get_multi_pointerto64_size(int32_t n){
  int ps = sizeof(int64_t) / sizeof(int32_t*);
  return (n + ps-1) / ps * ps;
}

}
}

#endif
