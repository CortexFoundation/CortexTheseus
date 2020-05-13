#include "cuda_ops.h"

namespace cvm {
namespace runtime{

inline __device__ int32_t broadcast_i_index(const int64_t* oshape, int o_index, const int64_t* ishape, int idim, int odim){
  int index = 0;
  int allIndex = 1;
  for(int i = 0; i < idim; i++){
    int idx = idim - 1 - i;
    int ovar = o_index % oshape[idx + odim-idim + MAX_DIM - odim];
    if(ovar < ishape[idx + MAX_DIM - idim]){
      index += allIndex * ovar;
    }
    allIndex = allIndex * ishape[idx + MAX_DIM - idim];
    o_index /= oshape[idx + odim-idim + MAX_DIM - odim];
  }
  return index;
}

template<typename F>
__global__ void kernel_broadcast(const int32_t *a, const int32_t *b, int32_t*c, 
    const int64_t n,
    int32_t adim, int32_t bdim, int32_t cdim,
    F const& f,
    const int64_t ashp0, const int64_t ashp1, const int64_t ashp2, const int64_t ashp3, const int64_t ashp4, const int64_t ashp5,
    const int64_t bshp0, const int64_t bshp1, const int64_t bshp2, const int64_t bshp3, const int64_t bshp4, const int64_t bshp5,
    const int64_t cshp0, const int64_t cshp1, const int64_t cshp2, const int64_t cshp3, const int64_t cshp4, const int64_t cshp5){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t ashape[MAX_DIM] = {ashp0, ashp1, ashp2, ashp3, ashp4, ashp5};
  const int64_t bshape[MAX_DIM] = {bshp0, bshp1, bshp2, bshp3, bshp4, bshp5};
  const int64_t cshape[MAX_DIM] = {cshp0, cshp1, cshp2, cshp3, cshp4, cshp5};
  for(uint64_t i = tid; i < n; i += gridDim.x * blockDim.x){
    int ai = broadcast_i_index(cshape, i, ashape, adim, cdim);
    int bi = broadcast_i_index(cshape, i, bshape, bdim, cdim);
    c[i] = f(a[ai], b[bi]);
  }
}

template<typename F>
__global__ void kernel_broadcast_scalar_a(const int32_t * a, const int32_t *b, int32_t *c, const int64_t n, F const& f){
  int lid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = lid; i < n; i+= gridDim.x * blockDim.x){
    c[i] = f(a[0], b[i]);
  }
}
template<typename F>
__global__ void kernel_broadcast_scalar_b(const int32_t * a, const int32_t *b, int32_t *c, const int64_t n, F const& f){
  int lid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = lid; i < n; i+= gridDim.x * blockDim.x){
    c[i] = f(a[i], b[0]);
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

  if(adim == 1 && ashape[0] == 1){
    kernel_broadcast_scalar_a<<<blockSize, threadSize>>>(a, b, c, n, f);
  }else if(bdim == 1 && bshape[0] == 1){
    kernel_broadcast_scalar_b<<<blockSize, threadSize>>>(a, b, c, n, f);
  }else{
    int64_t dev_ashape[MAX_DIM], dev_bshape[MAX_DIM], dev_cshape[MAX_DIM];
    get_cuda_shape(ashape, adim, dev_ashape);
    get_cuda_shape(bshape, bdim, dev_bshape);
    get_cuda_shape(cshape, cdim, dev_cshape);
    kernel_broadcast<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n, adim, bdim, cdim, f,
        dev_ashape[0], dev_ashape[1], dev_ashape[2], dev_ashape[3], dev_ashape[4], dev_ashape[5],
        dev_bshape[0], dev_bshape[1], dev_bshape[2], dev_bshape[3], dev_bshape[4], dev_bshape[5],
        dev_cshape[0], dev_cshape[1], dev_cshape[2], dev_cshape[3], dev_cshape[4], dev_cshape[5]);
  }
  return "";
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
}
}
