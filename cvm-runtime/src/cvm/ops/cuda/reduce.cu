#include "cuda_ops.h"

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
__global__ void kernel_reduce_with_axis(const int32_t *x, int32_t *y, const int32_t *realAxis,
    const int64_t *xshape, const int64_t *yshape, const int32_t axis_ndim, const uint64_t *every_xdim_size,
    const int32_t xndim, const int32_t yndim, const int64_t ysize, const int32_t* flag, const int64_t axis_size, F const& f){
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(uint64_t i =tid; i < ysize; i+= gridDim.x*blockDim.x){
    uint64_t in_i = 0, o_i = i;
    for(int j = yndim-1, xj = xndim-1; j>=0; j--){
      uint64_t col = o_i % yshape[j];
      o_i /= yshape[j];
      while(xj >= 0 && flag[xj--] == 1);
      in_i += col * every_xdim_size[xj+1];
    }
    int32_t tmp = x[in_i];
    for(uint64_t xi = 1; xi < axis_size; xi++){
      uint64_t o_i = xi, tmp_in_i = 0;
      for(int j = axis_ndim - 1; j>=0; j--){
        uint64_t col = o_i % xshape[realAxis[j]];
        o_i /= xshape[realAxis[j]];
        tmp_in_i += col * every_xdim_size[realAxis[j]];
      }
      tmp = f(tmp, x[in_i + tmp_in_i]);
    }
    y[i] = tmp;
  }
}
template<typename F>
const char* cuda_reduce(const int32_t *x, int32_t *y, const uint64_t xsize, const uint64_t ysize, const int64_t *xshape, const int64_t *yshape, const int32_t* realAxis, const int32_t* flag, const uint64_t *every_xdim_size, const int64_t axis_size,const int32_t xndim, const int32_t yndim, const int32_t axis_ndim, F const& f, int& error_code){
  int64_t *dev_xshape = NULL, *dev_yshape = NULL;
  uint64_t *dev_every_xdim_size = NULL;
  int32_t *dev_flag = NULL, *dev_axis = NULL;
  if(axis_ndim == 0){
    kernel_reduce<<<1, 256>>>(x, y, xsize, f);
    int error = cudaGetLastError();
    if(error != cudaSuccess){
        error_code = ERROR_KERNEL;
    }
  }else{
    int bSize = 256;
    int gSize = getGridSize(ysize, bSize);//(ysize + bSize - 1) / bSize;
    cudaError_t status;
    status = cudaMalloc((void**)&dev_xshape, sizeof(int64_t)*xndim);
    if(status != cudaSuccess){
      error_code = ERROR_MALLOC;
      goto end;
    }
    status = cudaMalloc((void**)&dev_yshape, sizeof(int64_t)*yndim);
    if(status != cudaSuccess){
      error_code = ERROR_MALLOC;
      goto end;
    }
    status = cudaMalloc((void**)&dev_axis, sizeof(int32_t) * axis_ndim);
    if(status != cudaSuccess){
      error_code = ERROR_MALLOC;
      goto end;
    }
    status = cudaMalloc((void**)&dev_every_xdim_size, sizeof(uint64_t) * xndim);
    if(status != cudaSuccess){
      error_code = ERROR_MALLOC;
      goto end;
    }
    status = cudaMalloc((void**)&dev_flag, sizeof(int32_t)*xndim);
    if(status != cudaSuccess){
      error_code = ERROR_MALLOC;
      goto end;
    }
    status = cudaMemcpy(dev_xshape, xshape, sizeof(int64_t)*xndim, cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
      error_code = ERROR_MEMCPY;
      goto end;
    }
    status = cudaMemcpy(dev_yshape, yshape, sizeof(int64_t)*yndim, cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
      error_code = ERROR_MEMCPY;
      goto end;
    }
    status = cudaMemcpy(dev_axis, realAxis, sizeof(int32_t)*axis_ndim, cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
      error_code = ERROR_MEMCPY;
      goto end;
    }
    status = cudaMemcpy(dev_every_xdim_size, every_xdim_size, sizeof(uint64_t) * xndim, cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
      error_code = ERROR_MEMCPY;
      goto end;
    }
    status = cudaMemcpy(dev_flag, flag, sizeof(int32_t)*xndim, cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
      error_code = ERROR_MEMCPY;
      goto end;
    }

    kernel_reduce_with_axis<<<gSize, bSize>>>(x, y, dev_axis, dev_xshape, dev_yshape, axis_ndim, 
        dev_every_xdim_size, xndim, yndim, ysize, dev_flag, axis_size, f);
    if(cudaSuccess != cudaGetLastError()){
        error_code = ERROR_KERNEL;
    }
  }
  print_to_file(y, ysize, "/tmp/zkh/trec/gpu/sum.txt");

end:
  if(dev_xshape != NULL) cudaFree(dev_xshape);
  if(dev_yshape != NULL) cudaFree(dev_yshape);
  if(dev_axis != NULL) cudaFree(dev_axis);
  if(dev_every_xdim_size != NULL) cudaFree(dev_every_xdim_size);
  if(dev_flag != NULL) cudaFree(dev_flag);
  return check_cuda_error(cudaGetLastError());
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

