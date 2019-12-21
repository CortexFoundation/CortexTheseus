#include "cuda_ops.h"

inline __device__ int32_t broadcast_i_index(int64_t* oshape, int o_index, int64_t* ishape, int idim, int odim){
  int index = 0;
  int allIndex = 1;
  for(int i = 0; i < idim; i++){
    int idx = idim - 1 - i;
    int ovar = o_index % oshape[idx + odim-idim];
    if(ovar < ishape[idx]){
      index += allIndex * ovar;
    }
    allIndex = allIndex * ishape[idx];
    o_index /= oshape[idx + odim-idim];
  }
  return index;
}

template<typename F>
__global__ void kernel_broadcast(const int32_t *a, const int32_t *b, int32_t*c, 
    const int64_t n,
    int64_t *ashape, int32_t adim,
    int64_t *bshape, int32_t bdim,
    int64_t *cshape, int32_t cdim,
    F const& f){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x * blockDim.x){
    int ai = broadcast_i_index(cshape, i, ashape, adim, cdim);
    int bi = broadcast_i_index(cshape, i, bshape, bdim, cdim);
    c[i] = f(a[ai], b[bi]);
  }
}

template<typename F>
const char* cuda_broadcast(const int32_t *a, const int32_t *b, int32_t* c, 
    const uint64_t n, 
    int64_t *ashape, int32_t adim,
    int64_t *bshape, int32_t bdim,
    int64_t *cshape, int32_t cdim,
    F const& f,
    int& error_code)
{
  const int32_t *dev_a = a, *dev_b = b;
  int32_t *dev_c = c;
  int threadSize = 256;
  int blockSize = getGridSize(n, threadSize);

  int64_t *dev_ashape = NULL, *dev_bshape = NULL, *dev_cshape = NULL;
  cudaError_t status;
  status = cudaMalloc((void**)&dev_ashape, sizeof(int64_t) * adim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_bshape, sizeof(int64_t) * bdim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_cshape, sizeof(int64_t) * cdim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMemcpy(dev_ashape, ashape, sizeof(int64_t) * adim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMemcpy(dev_bshape, bshape, sizeof(int64_t) * bdim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  status = cudaMemcpy(dev_cshape, cshape, sizeof(int64_t) * cdim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  kernel_broadcast<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n, dev_ashape, adim, dev_bshape, bdim, dev_cshape, cdim, f);
  //cudaDeviceSynchronize();

  status = cudaGetLastError();
  if(cudaSuccess != status){
    error_code = ERROR_KERNEL;
  }
end:
  if(dev_ashape != NULL) cudaFree(dev_ashape);
  if(dev_bshape != NULL) cudaFree(dev_bshape);
  if(dev_cshape != NULL) cudaFree(dev_cshape);
  return check_cuda_error(cudaGetLastError());
}

const char* cuda_broadcast_add(const int32_t *a, const int32_t *b, int32_t* c, 
    const uint64_t n, 
    int64_t *ashape, int32_t adim,
    int64_t *bshape, int32_t bdim,
    int64_t *cshape, int32_t cdim,
    int& error_code){
  auto f = []__device__(const int32_t a, const int32_t b){
    return a + b;
  };
  return cuda_broadcast(a, b, c, n, ashape, adim, bshape, bdim, cshape, cdim, f, error_code);
}

const char* cuda_broadcast_sub(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n, 
    int64_t *ashape, int32_t adim,
    int64_t *bshape, int32_t bdim,
    int64_t *cshape, int32_t cdim,
    int& error_code){
  auto f = []__device__(const int32_t a, const int32_t b){
    return a - b;
  };
  return cuda_broadcast(a, b, c, n, ashape, adim, bshape, bdim, cshape, cdim, f, error_code);
}

const char* cuda_broadcast_mul(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n, 
    int64_t *ashape, int32_t adim,
    int64_t *bshape, int32_t bdim,
    int64_t *cshape, int32_t cdim,
    int& error_code){
  auto f = []__device__(const int32_t a, const int32_t b){
    return a * b;
  };
  return cuda_broadcast(a, b, c, n, ashape, adim, bshape, bdim, cshape, cdim, f, error_code);
}

const char* cuda_broadcast_max(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n, 
    int64_t *ashape, int32_t adim,
    int64_t *bshape, int32_t bdim,
    int64_t *cshape, int32_t cdim,
    int& error_code){
  auto f = []__device__(const int32_t a, const int32_t b){
    return a > b ? a : b;
  };
  return cuda_broadcast(a, b, c, n, ashape, adim, bshape, bdim, cshape, cdim, f, error_code);
}

const char* cuda_broadcast_div(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n, 
    int64_t *ashape, int32_t adim,
    int64_t *bshape, int32_t bdim,
    int64_t *cshape, int32_t cdim,
    int& error_code){
  auto f = []__device__(const int32_t a, const int32_t b){
    return b == 0 ? 0 : a/b;
  };
  return cuda_broadcast(a, b, c, n, ashape, adim, bshape, bdim, cshape, cdim, f, error_code);
}

const char* cuda_broadcast_greater(const int32_t *a, const int32_t *b, int32_t* c, const uint64_t n, 
    int64_t *ashape, int32_t adim,
    int64_t *bshape, int32_t bdim,
    int64_t *cshape, int32_t cdim,
    int& error_code){
  auto f = []__device__(const int32_t a, const int32_t b){
    return a > b;
  };
  return cuda_broadcast(a, b, c, n, ashape, adim, bshape, bdim, cshape, cdim, f, error_code);
}
