#include "cuda_ops.h"

namespace cvm{
namespace runtime{

template<typename F>
__global__ void kernel_reduce(const int32_t *x, int32_t *y, int64_t n, F const& f){
  __shared__ int32_t buf[256];
  int32_t tid = threadIdx.x;
  int32_t tmp = 0;
  if(tid < n){
    tmp = x[tid];
  }
  for (int i = tid + blockDim.x; i < n; i += blockDim.x){
    tmp = f(tmp, x[i]);
  }

  buf[tid] = tmp;
  __syncthreads();
  for(int s = 1; s < blockDim.x && s < n; s*=2){
    if((tid % (2*s)) == 0){
      buf[tid] = f(buf[tid], buf[tid + s]);
    }
    __syncthreads();
  }

  if(tid == 0) y[0] = buf[0];
}

template<typename F>
__global__ void kernel_reduce_with_axis(const int32_t *x, int32_t *y, 
    const int32_t axis_ndim, const int32_t xndim, const int32_t yndim, const int64_t ysize, const int64_t axis_size, F const& f,
    const int64_t xshp0, const int64_t xshp1, const int64_t xshp2, const int64_t xshp3, const int64_t xshp4, const int64_t xshp5,
    const int64_t yshp0, const int64_t yshp1, const int64_t yshp2, const int64_t yshp3, const int64_t yshp4, const int64_t yshp5, 
    const int32_t axshp0, const int32_t axshp1, const int32_t axshp2, const int32_t axshp3, const int32_t axshp4, const int32_t axshp5,
    const uint64_t exshp0, const uint64_t exshp1, const uint64_t exshp2, const uint64_t exshp3, const uint64_t exshp4, const uint64_t exshp5,
    const int32_t fshp0, const int32_t fshp1, const int32_t fshp2, const int32_t fshp3, const int32_t fshp4, const int32_t fshp5){
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int64_t xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int64_t yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  const int32_t realAxis[MAX_DIM] = {axshp0, axshp1, axshp2, axshp3, axshp4, axshp5};
  const uint64_t every_xdim_size[MAX_DIM] = {exshp0, exshp1, exshp2, exshp3, exshp4, exshp5};
  const int32_t flag[MAX_DIM] = {fshp0, fshp1, fshp2, fshp3, fshp4, fshp5};

  for(uint64_t i =tid; i < ysize; i+= gridDim.x*blockDim.x){
    uint64_t in_i = 0, o_i = i;
    for(int j = yndim-1, xj = xndim-1; j>=0; j--){
      uint64_t col = o_i % yshape[j + MAX_DIM - yndim];
      o_i /= yshape[j + MAX_DIM - yndim];
      while(xj >= 0 && flag[(xj--) + MAX_DIM - xndim] == 1);
      in_i += col * every_xdim_size[xj+1 + MAX_DIM - xndim];
    }
    int32_t tmp = x[in_i];
    for(uint64_t xi = 1; xi < axis_size; xi++){
      uint64_t o_i = xi, tmp_in_i = 0;
      for(int j = axis_ndim - 1; j>=0; j--){
        uint64_t col = o_i % xshape[realAxis[j + MAX_DIM - axis_ndim] + MAX_DIM - xndim];
        o_i /= xshape[realAxis[j + MAX_DIM - axis_ndim] + MAX_DIM - xndim];
        tmp_in_i += col * every_xdim_size[realAxis[j + MAX_DIM - axis_ndim] + MAX_DIM - xndim];
      }
      tmp = f(tmp, x[in_i + tmp_in_i]);
    }
    y[i] = tmp;
  }
}
template<typename F>
const char* cuda_reduce(const int32_t *x, int32_t *y, const uint64_t xsize, const uint64_t ysize, const int64_t *xshape, const int64_t *yshape, const int32_t* realAxis, const int32_t* flag, const uint64_t *every_xdim_size, const int64_t axis_size,const int32_t xndim, const int32_t yndim, const int32_t axis_ndim, F const& f, int& error_code){
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM];
  uint64_t dev_every_xdim_size[MAX_DIM];
  int32_t dev_flag[MAX_DIM], dev_axis[MAX_DIM];
  if(axis_ndim == 0){
    kernel_reduce<<<1, 256>>>(x, y, xsize, f);
    int error = cudaGetLastError();
    if(error != cudaSuccess){
        error_code = ERROR_KERNEL;
    }
  }else{
    int bSize = 256;
    int gSize = getGridSize(ysize, bSize);//(ysize + bSize - 1) / bSize;
    get_cuda_shape(xshape, xndim, dev_xshape);
    get_cuda_shape(yshape, yndim, dev_yshape);
    get_cuda_shape(realAxis, axis_ndim, dev_axis);
    get_cuda_shape(every_xdim_size, xndim, dev_every_xdim_size);
    get_cuda_shape(flag, xndim, dev_flag);

    kernel_reduce_with_axis<<<gSize, bSize>>>(x, y, axis_ndim, 
         xndim, yndim, ysize, axis_size, f,
      dev_xshape[0], dev_xshape[1], dev_xshape[2], dev_xshape[3], dev_xshape[4], dev_xshape[5],
      dev_yshape[0], dev_yshape[1], dev_yshape[2], dev_yshape[3], dev_yshape[4], dev_yshape[5],
      dev_axis[0], dev_axis[1], dev_axis[2], dev_axis[3], dev_axis[4], dev_axis[5],
      dev_every_xdim_size[0], dev_every_xdim_size[1], dev_every_xdim_size[2], dev_every_xdim_size[3], dev_every_xdim_size[4], dev_every_xdim_size[5],
      dev_flag[0], dev_flag[1], dev_flag[2], dev_flag[3], dev_flag[4], dev_flag[5]);
  }
  print_to_file(y, ysize, "sum.txt");

  return "";//check_cuda_error(cudaGetLastError());
}
const char* cuda_sum(const int32_t *x, int32_t *y, const uint64_t xsize, 
    const uint64_t ysize,  const int64_t *xshape, const int64_t *yshape, 
    const int32_t* realAxis, const int32_t* flag, const uint64_t *every_xdim_size, 
    const int64_t axis_size,const int32_t xndim, const int32_t yndim,
    const int32_t axis_ndim, int& error_code){
  auto f = []__device__(const int32_t a, const int32_t b){
    return a + b; 
  };
  return cuda_reduce(x, y, xsize, ysize, xshape, yshape, realAxis, flag, every_xdim_size, axis_size, xndim, yndim, axis_ndim, f, error_code);
}

const char* cuda_max(const int32_t *x, int32_t *y, const uint64_t xsize, const uint64_t ysize,
    const int64_t *xshape, const int64_t *yshape, const int32_t* realAxis, const int32_t* flag, 
    const uint64_t *every_xdim_size, const int64_t axis_size,
    const int32_t xndim, const int32_t yndim, const int32_t axis_ndim, int& error_code){
  auto f = []__device__(const int32_t a, const int32_t b){
    return a > b ? a : b;
  };
  return cuda_reduce(x, y, xsize, ysize, xshape, yshape, realAxis, flag, every_xdim_size, axis_size, xndim, yndim, axis_ndim, f, error_code);
}

}
}
