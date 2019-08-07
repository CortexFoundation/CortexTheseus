#include "cuda_ops.h"

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

__global__ void kernel_clip(const int32_t *x, int32_t *y,
    const uint64_t n, const int32_t maxV, const int32_t minV){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x*blockDim.x){
    y[i] = max(min(x[i], maxV), minV);
  }
}

const char* cuda_clip(const int32_t *x, int32_t *y, const uint64_t n, const int32_t max, const int32_t min, int& error_code){
  int threadSize = 256;
  int blockSize = getGridSize(n, threadSize);
  kernel_clip<<<blockSize, threadSize>>>(x, y, n, max, min);
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}

const char* cuda_cvm_clip(const int32_t* x, const int32_t precision, int32_t *y, const uint64_t n, int& error_code){
  int32_t min = -(((int64_t)1 << (precision-1))-1);
  int32_t max = -min;
  int bSize = 256;
  int gSize = getGridSize(n, bSize); 
  kernel_clip<<<gSize, bSize>>>(x, y, n, max, min);
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  print_to_file(y, n, "/tmp/zkh/trec/gpu/cvm_clip.txt");
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
  int32_t minV = -(((int64_t)1 << (precision - 1)) - 1);
  int32_t maxV = -minV;
  for(uint64_t i = tid; i < n; i+= gridDim.x*blockDim.x){
    int shift_a = a[i];
    if(b == 0) c[i] = shift_a;
    else {
      shift_a = ((shift_a >> (b - 1)) + 1 ) >> 1;
      c[i] = max(min(shift_a, maxV), minV);
    } 
  }
}
const char* cuda_cvm_right_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const uint64_t n, int& error_code){
  int bSize = 256;
  int gSize = getGridSize(n, bSize);
  kernel_cvm_right_shift<<<gSize, bSize>>>(a, b, precision, c, n);
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
