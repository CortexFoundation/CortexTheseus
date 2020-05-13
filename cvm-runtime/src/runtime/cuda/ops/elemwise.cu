#include "./cuda_ops.h"

namespace cvm{
namespace runtime{

template<typename F>
__global__ void kernel_elemwise(const int32_t *a, const int32_t *b, int32_t *c, uint64_t n, F const &op){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x * blockDim.x){
    c[i] = op(a[i], b[i]);
  }
}

template<typename F>
const char* cuda_elemwise(const int32_t *a, const int32_t *b, int32_t *c, uint64_t n, F const &f, int& error_code){
  int blockSize = 256;
  int gridSize = getGridSize(n, blockSize);
  kernel_elemwise<<<gridSize, blockSize>>>(a, b, c, n, f);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}

const char* cuda_elemwise_add(int32_t *a, int32_t *b, int32_t *c, uint64_t n, int& error_code){
  const auto& f = []__device__(const int32_t a, const int32_t b){
    return a + b;
  };
  return cuda_elemwise(a, b, c, n, f, error_code);
}

const char* cuda_elemwise_sub(int32_t *a, int32_t *b, int32_t *c, uint64_t n, int& error_code){
  cudaError_t error = cudaGetLastError();
  auto f = []__device__(const int32_t a, const int32_t b){
    return a - b;
  };
  return cuda_elemwise(a, b, c, n, f, error_code);
}

__device__ __forceinline__ int4 d_clip(int4 data, int maxV, int minV){
    data.x = max(min(data.x, maxV), minV);
    data.y = max(min(data.y, maxV), minV);
    data.z = max(min(data.z, maxV), minV);
    data.w = max(min(data.w, maxV), minV);
    return data;
}
__global__ void kernel_clip(const int32_t *x, int32_t *y,
    const uint64_t n, const int32_t maxV, const int32_t minV){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x*blockDim.x){
    y[i] = max(min(x[i], maxV), minV);
  }
}
__global__ void kernel_clip4(const int4 *x, int4 *y,
    const uint64_t n, const int32_t maxV, const int32_t minV){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x*blockDim.x){
    int4 data = x[i];
    y[i] = d_clip(data, maxV, minV);
  }
}

const char* cuda_clip(const int32_t *x, int32_t *y, const uint64_t n, const int32_t max, const int32_t min, int& error_code){
  int new_n = n / 4;
  int threadSize = 256;
  int blockSize = getGridSize(new_n, threadSize);
  if(new_n > 0)
    kernel_clip4<<<blockSize, threadSize>>>((int4*)x, (int4*)y, new_n, max, min);
  kernel_clip<<<1, 32>>>(x+new_n*4, y+new_n*4, n-new_n*4, max, min);
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}

const char* cuda_cvm_clip(const int32_t* x, const int32_t precision, int32_t *y, const uint64_t n, int& error_code){
  int32_t min = -(((int64_t)1 << (precision-1))-1);
  int32_t max = -min;
  int new_n = n / 4;
  int bSize = 256;
  int gSize = getGridSize(new_n, bSize); 
  if(new_n > 0)
    kernel_clip4<<<gSize, bSize>>>((int4*)x, (int4*)y, new_n, max, min);
  kernel_clip<<<1, 32>>>(x+new_n*4, y+new_n*4, n-new_n*4, max, min);
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  print_to_file(y, n, "cvm_clip.txt");
  return check_cuda_error(error);
}


const char* cuda_flatten(const int32_t *x, int32_t *y, const uint64_t n, int& error_code){
  if(x == y) return NULL;
  cudaMemcpy(y, x, n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_MEMCPY;
  }
  return check_cuda_error(error);
}

const char* cuda_reshape(const int32_t *x, int32_t *y, uint64_t n, int& error_code){
  if(x == y) return NULL;
  cudaMemcpy(y, x, n*sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess){
    error_code = ERROR_MEMCPY;
  }
  return check_cuda_error(error);
}

__global__ void kernel_cvm_right_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const uint64_t n){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t minV = -(((int64_t)1 << (precision - 1)) - 1);
  const int32_t maxV = -minV;
  for(uint64_t i = tid; i < n; i+= gridDim.x*blockDim.x){
    int shift_a = a[i];
    shift_a = ((shift_a >> (b - 1)) + 1 ) >> 1;
    c[i] = max(min(shift_a, maxV), minV);
  }
}
__global__ void kernel_cvm_right_shift4(const int4 *a, const int32_t b, const int32_t precision, int4 *c, const uint64_t n){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t minV = -(((int64_t)1 << (precision - 1)) - 1);
  const int32_t maxV = -minV;
  for(uint64_t i = tid; i < n; i+= gridDim.x*blockDim.x){
    int4 shift_a = a[i];
    shift_a.x = ((shift_a.x >> (b - 1)) + 1 ) >> 1;
    shift_a.y = ((shift_a.y >> (b - 1)) + 1 ) >> 1;
    shift_a.z = ((shift_a.z >> (b - 1)) + 1 ) >> 1;
    shift_a.w = ((shift_a.w >> (b - 1)) + 1 ) >> 1;

    c[i] = d_clip(shift_a, maxV, minV);
  }
}
const char* cuda_cvm_right_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const uint64_t n, int& error_code){
  if(b==0){
    cudaMemcpy(c, a, n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  }else{
    int new_n = n/4;
    int bSize = 256;
    int gSize = getGridSize(new_n, bSize);
    if(new_n > 0)
      kernel_cvm_right_shift4<<<gSize, bSize>>>((int4*)a, b, precision, (int4*)c, new_n);
    kernel_cvm_right_shift<<<1, 32>>>(a+new_n*4, b, precision, c+new_n*4, n-new_n*4);
  }
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}

__global__ void kernel_cvm_left_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const uint64_t n){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t minV = -(((int64_t)1 << (precision - 1)) - 1);
  int32_t maxV = -minV;
  for(uint64_t i = tid; i < n; i += gridDim.x*blockDim.x){
    int32_t shift_a = a[i] << b;
    c[i] = max(min(shift_a, maxV), minV);
  }
}
const char* cuda_cvm_left_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const uint64_t n, int& error_code){
  int bSize = 256;
  int gSize = getGridSize(n, bSize);//(n + bSize - 1) / bSize;
  kernel_cvm_left_shift<<<gSize, bSize>>>(a, b, precision, c, n);
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}
}
}
