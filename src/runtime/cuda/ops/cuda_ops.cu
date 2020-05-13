#include "cuda_ops.h"

namespace cvm{
namespace runtime{
#define TILE_WIDTH 8
  
  
template<bool useBias>
__global__ void kernel_dense(
    const int32_t * __restrict__ A, // m*k 
    const int32_t * __restrict__ B, // was transposed, n*k
    int32_t *C, // m*n
    int32_t m, int32_t k, int32_t n, int32_t *bias){
  __shared__ int32_t sharedM[TILE_WIDTH][TILE_WIDTH];
  __shared__ int32_t sharedN[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = by*TILE_WIDTH + ty;
  int col = bx*TILE_WIDTH + tx;
  int sum = 0;

  for (int i = 0; i < (int)(ceil((float)k/TILE_WIDTH)); i++)
  {
    if (i*TILE_WIDTH + tx < k && row < m)//m*k
      sharedM[ty][tx] = A[row*k + i*TILE_WIDTH + tx];
    else
      sharedM[ty][tx] = 0;

    if(i*TILE_WIDTH + tx < k && bx*TILE_WIDTH + ty < n)//n*k
      //sharedN[tx][ty] = B[col * k + i * TILE_WIDTH + ty];
      sharedN[ty][tx] = B[(bx*TILE_WIDTH + ty) * k + i * TILE_WIDTH + tx];
    else
      sharedN[ty][tx] = 0;
    __syncthreads();

    for(int j = 0; j < TILE_WIDTH; j++)
      sum += sharedM[ty][j] * sharedN[tx][j];
    __syncthreads();
  }
  if (row < m && col < n){
    if(useBias) sum += bias[col];
    C[row*n + col] = sum;
  }
}

const char* cuda_dense(
    int32_t *a,
    int32_t *b,
    int32_t *c,
    const int m, const int k, const int n, int32_t* bias, int& error_code){
  int32_t *dev_a = a, *dev_b = b, *dev_c = c, *dev_bias = bias;
  printf("%d %d %d\n", m, k, n);

  dim3 bDim(TILE_WIDTH, TILE_WIDTH, 1);
  int gh = (m + TILE_WIDTH - 1) / TILE_WIDTH;
  int gw = (n + TILE_WIDTH - 1) / TILE_WIDTH;
  dim3 gDim(gw, gh, 1);
  if(bias != NULL)
    kernel_dense<true><<<gDim, bDim>>>(dev_a, dev_b, dev_c, m, k, n, dev_bias);
  else
    kernel_dense<false><<<gDim, bDim>>>(dev_a, dev_b, dev_c, m, k, n, dev_bias);

  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  print_to_file(dev_c, m*n, "dense.txt");
  return check_cuda_error(error);
}

__global__ void kernel_relu(const int32_t *x, int32_t*y, const uint64_t n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x * blockDim.x){
    y[i] = max(x[i], 0);
  }
}
__global__ void kernel_relu4(const int4 *x, int4 *y, const uint64_t n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x * blockDim.x){
    int4 data = x[i];
    data.x = max(data.x, 0);
    data.y = max(data.y, 0);
    data.z = max(data.z, 0);
    data.w = max(data.w, 0);
    y[i] = data; 
  }
}
const char* cuda_relu(const int32_t *x, int32_t *y, const uint64_t n, int& error_code){
  const int32_t *dev_x = x;
  int32_t *dev_y = y;

  int new_n = n/4;
  int threadSize = 256;
  int blockSize = getGridSize(new_n, threadSize);
  if(new_n > 0)
    kernel_relu4<<<blockSize, threadSize>>>((int4*)dev_x, (int4*)dev_y, new_n);
  kernel_relu<<<1, 32>>>(dev_x+new_n*4, dev_y+new_n*4, n-new_n*4);

  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}

__global__ void kernel_log(const int32_t *x, int32_t *y, const uint64_t n){
  int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(uint64_t j = tid; j < n; j += gridDim.x * blockDim.x){
    const int64_t x_val = x[j];
    y[j] = 64;
    for(int i = 1; i < 64; i++){
      int64_t tmp = (int64_t)1 << i;
      if(abs(x_val) < tmp){
        y[j] = i;
        break;
      }
    }
  }
}
const char* cuda_log(const int32_t *x, int32_t *y, const uint64_t n, int& error_code){
  const int32_t *dev_x = x;
  int32_t *dev_y = y;

  int bSize = 256;
  int gSize = getGridSize(n, bSize);
  kernel_log<<<gSize,bSize>>>(dev_x, dev_y, n);

  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}
__global__ void kernel_abs(const int32_t *x, int32_t *y, const uint64_t n){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x*blockDim.x){
    y[i] = abs(x[i]);
  }
}
const char* cuda_abs(const int32_t *x, int32_t *y, const uint64_t n, int& error_code){
  const int32_t *dev_x = x;
  int32_t *dev_y = y;
  int bSize = 256;
  int gSize = getGridSize(n, bSize);//(n + bSize - 1) / bSize;
  kernel_abs<<<gSize, bSize>>>(dev_x, dev_y, n);
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}

/* __global__ void kernel_sqrt(const int32_t *x, int32_t *y, const uint64_t n){ */
  /* int tid = threadIdx.x + blockDim.x * blockIdx.x; */
  /* for(uint64_t i = tid; i < n; i += gridDim.x*blockDim.x){ */
    /* y[i] = x[i] < 0 ? 0 : static_cast<int32_t>(sqrt(static_cast<double>(x[i]))); */
  /* } */
/* } */
/* const char* cuda_sqrt(const int32_t *x, int32_t *y, const uint64_t n, int& error_code){ */
  /* const int32_t *dev_x = x; */
  /* int32_t *dev_y = y; */
  /* int bSize = 256; */
  /* int gSize = getGridSize(n, bSize);//(n + bSize - 1) / bSize; */
  /* kernel_sqrt<<<gSize, bSize>>>(dev_x, dev_y, n); */
  /* cudaError_t error = cudaGetLastError(); */
  /* if(cudaSuccess != error){ */
    /* error_code = ERROR_KERNEL; */
  /* } */
  /* return check_cuda_error(error); */
/* } */

__global__ void kernel_concatenate(int32_t **input, const int64_t *ishapes, const int32_t ndim, int32_t *out_data, const int32_t axis, const int64_t *axisSize, const int64_t oshape0, const int64_t oshape1, const int64_t oshape2, const int64_t oshape3, const int64_t oshape4, const int64_t oshape5){
  int32_t bid = blockIdx.x;
  int32_t lid = threadIdx.x;
  const int32_t *input_data = input[bid];
  const int64_t *ishape = ishapes + bid * ndim;
  __shared__ int64_t share_ishape[MAX_DIM];
  if(lid < ndim) share_ishape[lid] = ishape[lid];
  __syncthreads();
  int64_t reg_ishape[MAX_DIM];
  int64_t isize = 1;
  for(int i = 0; i < ndim; i++){
    reg_ishape[i] = share_ishape[i];
    isize *= reg_ishape[i];
  }

  const int32_t y_axis_size = axisSize[bid];
  const int64_t oshape[MAX_DIM] = {oshape0, oshape1, oshape2, oshape3, oshape4, oshape5};
  for(int64_t i = lid; i < isize; i+= blockDim.x){
    int32_t tmp_i = i, yi = 0, shape_size = 1;
    for(int32_t d = ndim-1; d>=0; d--){
      int32_t col = tmp_i % reg_ishape[d];
      tmp_i /= reg_ishape[d];
      if(d == axis) col += y_axis_size;
      yi += col * shape_size;
      shape_size *= oshape[d + 6-ndim];
    }
    out_data[yi] = input_data[i];
  }
}

__global__ void kernel_concatenate_one_input(int32_t *input, const int32_t isize, const int32_t ndim, int32_t *out_data, const int32_t axis, const int32_t axisSize, 
    const int32_t ishp0, const int32_t ishp1, const int32_t ishp2, const int32_t ishp3, const int32_t ishp4, const int32_t ishp5,
    const int32_t oshp0, const int32_t oshp1, const int32_t oshp2, const int32_t oshp3, const int32_t oshp4, const int32_t oshp5){
  int32_t lid = threadIdx.x + blockIdx.x * blockDim.x;

  //const int32_t y_axis_size = axisSize;
  const int32_t ishape[MAX_DIM] = {ishp0, ishp1, ishp2, ishp3, ishp4, ishp5};
  const int32_t oshape[MAX_DIM] = {oshp0, oshp1, oshp2, oshp3, oshp4, oshp5};
  int32_t axis_shape[MAX_DIM] = {0};
  axis_shape[axis] = axisSize;
  for(int32_t i = lid; i < isize; i+= gridDim.x * blockDim.x){
    int32_t tmp_i = i, yi = 0, shape_size = 1;
    for(int32_t d = ndim-1; d>=0; d--){
      int32_t col = tmp_i % ishape[d + MAX_DIM - ndim];
      tmp_i /= ishape[d + MAX_DIM - ndim];
      col += axis_shape[d];
      yi += col * shape_size;
      shape_size *= oshape[d + MAX_DIM-ndim];
    }
    out_data[yi] = input[i];
  }
}

__global__ void kernel_concatenate_opt(const int32_t *input, int32_t *output, int y_axis_batch, int x_axis_batch, int n){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < n){
    int y_iter = tid / x_axis_batch;
    int x_iter = tid % x_axis_batch;
    output[y_iter * y_axis_batch+ x_iter] = input[y_iter * x_axis_batch + x_iter];
  }
}

const char* cuda_concatenate(int32_t **inputs, int64_t *ishapes, const int32_t ninput, const int32_t ndim, int32_t *output,const int64_t* oshape, const int32_t axis, int32_t* axisSize, int32_t *ext_space, int& error_code){
  int32_t *dev_output = output;
  int64_t dev_oshape[6];
  get_cuda_shape(oshape, ndim, dev_oshape);

  int64_t y_size = 1;
  for (int i = 0; i < axis; ++i) y_size *= oshape[i];
  int32_t axis_batch = 1;
  for (int i = axis+1; i < ndim; ++i) axis_batch *= oshape[i];

  int64_t y_start_idx = 0;
  int64_t y_axis_batch = oshape[axis] * axis_batch;
  for (int m = 0; m < ninput; ++m) {
    int32_t* Ix = inputs[m];
    int32_t x_axis_batch = ishapes[m*ndim+axis] * axis_batch;

    int32_t n = x_axis_batch * y_size;
    int32_t bSize = 512;
    int32_t gSize = getGridSize(n, bSize);
    kernel_concatenate_opt<<<gSize, bSize>>>(Ix, dev_output + y_start_idx, y_axis_batch, x_axis_batch, n);
    y_start_idx += x_axis_batch;
  }

  return "";
}

__global__ void kernel_bias_add(const int32_t *x_data, const int32_t * bias_data, int32_t *y_data, 
    int64_t ysize, const int64_t *yshape, const int32_t ndim, const int32_t axis){
  int32_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < ysize){
    int32_t bV = 0;
    for(int32_t j = ndim - 1; j >= 0; j--){
      if(j == axis){
        bV = bias_data[axis];
        break;
      }
    }
    y_data[i] = x_data[i] + bV;
  }
}
const char* cuda_bias_add(const int32_t *x_data, const int32_t * bias_data, int32_t *y_data, 
    int64_t ysize, const int64_t *yshape, const int32_t ndim, const int32_t axis, int& error_code){
  int64_t *dev_yshape;
  cudaMalloc((void**)&dev_yshape, sizeof(int64_t) * ndim);
  cudaMemcpy(dev_yshape, yshape, sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);

  int bSize = 256;
  int gSize = (ysize + bSize - 1) / bSize;
  kernel_bias_add<<<gSize, bSize>>>(x_data, bias_data, y_data, ysize, dev_yshape, ndim, axis);

  cudaFree(dev_yshape);
  return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_repeat(const int32_t *x_data, int32_t *y_data, const uint64_t ysize, const int32_t ndim, const int32_t axis, 
    const int32_t repeat,
    const int64_t xshp0, const int64_t xshp1, const int64_t xshp2, const int64_t xshp3, const int64_t xshp4, const int64_t xshp5,
    const int64_t yshp0, const int64_t yshp1, const int64_t yshp2, const int64_t yshp3, const int64_t yshp4, const int64_t yshp5){
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int64_t xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int64_t yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){
    uint64_t o_i = i, in_i = 0, shapeSize = 1;
    for(int j = ndim-1; j >= 0; j--){
      uint64_t col = o_i % yshape[j + MAX_DIM - ndim];
      o_i /= yshape[j + MAX_DIM - ndim];
      if(j == axis) col = col / repeat;
      in_i += col * shapeSize;
      shapeSize = shapeSize * xshape[j + MAX_DIM - ndim];
    }
    y_data[i] = x_data[in_i];
  }
}
const char* cuda_repeat(const int32_t *x_data, int32_t *y_data, const int64_t *xshape,
    const int64_t *yshape, const uint64_t ysize, const int32_t xndim, const int32_t yndim, 
    const int32_t axis, const int32_t repeat, int& error_code){
  int bSize = 256;
  int gSize = getGridSize(ysize, bSize);
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM];
  get_cuda_shape(xshape, xndim, dev_xshape);
  get_cuda_shape(yshape, yndim, dev_yshape);

  kernel_repeat<<<gSize, bSize>>>(x_data, y_data, ysize, yndim, axis, repeat,
      dev_xshape[0], dev_xshape[1], dev_xshape[2], dev_xshape[3], dev_xshape[4], dev_xshape[5],
      dev_yshape[0], dev_yshape[1], dev_yshape[2], dev_yshape[3], dev_yshape[4], dev_yshape[5]);

  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_upsampling_nearest(const int32_t *x_data, int32_t *y_data, const uint32_t scale, const uint32_t ih, const uint32_t iw,
    const uint32_t oh, const uint32_t ow, const uint32_t channel){
  int ox = threadIdx.x;
  int oy = threadIdx.y;
  
  for(int b = blockIdx.x; b < channel; b+=gridDim.x){
    for(int r = oy; r < oh; r += blockDim.y){
      for(int c = ox; c < ow; c += blockDim.x){
        y_data[b * oh * ow + r * ow + c] = x_data[b * ih * iw + r/scale * iw + c/scale];
      }
    }
  }
}

const char* cuda_upsampling_nearest(const int32_t *x_data, int32_t *y_data, const uint32_t scale, const int32_t ih, const int32_t iw, 
    const uint32_t oh, const uint32_t ow, const uint32_t batch, const uint32_t channel, int& error_code){
  dim3 block(1, 32, 32);
  int grid = channel > 4096 ? 4096 : channel;

  for(uint32_t i = 0; i < batch; i++){
    kernel_upsampling_nearest<<<grid, block>>>(x_data + i*channel*ih*iw, 
        y_data + i*channel*oh*ow, 
        scale, ih, iw, oh, ow, channel);
    if(cudaSuccess != cudaGetLastError()){
        error_code = ERROR_KERNEL;
        return check_cuda_error(cudaGetLastError());
    }
  }
  return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_negative(const int32_t *x_data, int32_t *y_data, uint64_t n){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x*blockDim.x){
    y_data[i] = -x_data[i];
  }
}
const char* cuda_negative(const int32_t *x_data, int32_t *y_data, uint64_t n, int& error_code){
  int threadSize = 256;
  int blockSize = getGridSize(n, threadSize);
  kernel_negative<<<blockSize, threadSize>>>(x_data, y_data, n);
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}


__global__ void kernel_tile(const int32_t *x_data, int32_t *y_data, const uint64_t ysize, const int32_t yndim, const int32_t xndim,
    const int64_t xshp0, const int64_t xshp1, const int64_t xshp2, const int64_t xshp3, const int64_t xshp4, const int64_t xshp5,
    const int64_t yshp0, const int64_t yshp1, const int64_t yshp2, const int64_t yshp3, const int64_t yshp4, const int64_t yshp5){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int64_t yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){
    uint64_t o_i = i, in_i = 0, shapeSize = 1;
    for(int j = xndim-1; j >= 0; j--){
      int yj = j + yndim - xndim;
      int col = o_i % yshape[yj + MAX_DIM - yndim];
      o_i /= yshape[yj + MAX_DIM - yndim];
      col = col % xshape[j + MAX_DIM - xndim];
      in_i += col * shapeSize;
      shapeSize = shapeSize * xshape[j + MAX_DIM - xndim];
    }
    y_data[i] = x_data[in_i];
  }
}
const char* cuda_tile(const int32_t *x_data, int32_t *y_data, const uint64_t ysize, const int32_t yndim, const int32_t xndim,
    const int64_t *xshape, const int64_t *yshape, int& error_code){
  uint64_t tmp_y_size = 1;
  for(int i = 0; i < xndim; i++){
    tmp_y_size *= yshape[i + yndim - xndim];
  }
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM];
  get_cuda_shape(xshape, xndim, dev_xshape);
  get_cuda_shape(yshape, yndim, dev_yshape);

  int threadSize = 256;
  int blockSize = getGridSize(tmp_y_size, threadSize);//(tmp_y_size + threadSize - 1) / threadSize;
  uint64_t othery = 1;
  cudaError_t status;

  kernel_tile<<<blockSize, threadSize>>>(x_data, y_data, tmp_y_size, yndim, xndim,
      dev_xshape[0], dev_xshape[1], dev_xshape[2], dev_xshape[3], dev_xshape[4], dev_xshape[5],
      dev_yshape[0], dev_yshape[1], dev_yshape[2], dev_yshape[3], dev_yshape[4], dev_yshape[5]);

  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
    goto end;
  }
  for(int i = 0; i < yndim-xndim; i++){
    othery *= yshape[i];
  }
  for(size_t i = 1; i < othery; i++){
    status = cudaMemcpy(y_data + i*tmp_y_size, y_data, tmp_y_size * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    if(status != cudaSuccess){
      error_code = ERROR_MEMCPY;
      goto end;
    }
  }

end:
  return check_cuda_error(cudaGetLastError());
}

/* __global__ void kernel_pad(const int32_t *x_data, int32_t *y_data, const uint64_t ndim, const int32_t ysize, */
    /* const int32_t xshp0, const int32_t xshp1, const int32_t xshp2, const int32_t xshp3, const int32_t xshp4, const int32_t xshp5, */
    /* const int32_t yshp0, const int32_t yshp1, const int32_t yshp2, const int32_t yshp3, const int64_t yshp4, const int64_t yshp5, */
    /* const int32_t pad0, const int32_t pad1, const int32_t pad2, const int32_t pad3, const int32_t pad4, const int32_t pad5, */
    /* const int32_t pad_value){ */
  /* int tid = threadIdx.x + blockIdx.x * blockDim.x; */
  /* const int64_t xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5}; */
  /* const int64_t yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5}; */
  /* const int32_t pad_before[MAX_DIM] = {pad0, pad1, pad2, pad3, pad4, pad5}; */
  /* for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){ */
    /* uint64_t o_i = i, in_i = 0, shapeSize = 1; */
    /* bool flag = true; */
    /* for(int j = ndim-1; j >= 0; j--){ */
      /* int sj = j+MAX_DIM-ndim; */
      /* int col = o_i % yshape[sj]; */
      /* int lower = pad_before[sj], upper = pad_before[sj]+pad_before[sj]; */
      /* if (col < lower || col >= upper) { */
        /* flag = false; */
        /* break; */
      /* } */
      /* o_i /= yshape[j]; */
      /* in_i += (col-lower) * shapeSize; */
      /* shapeSize *=  xshape[sj]; */
    /* } */
    /* y_data[i] = flag ? x_data[in_i] : pad_value; */
  /* } */
/* } */
/* const char* cuda_pad(const int32_t *x_data, const int32_t *pad_data, const int32_t pad_value, int32_t *y_data, */
    /* const int64_t *xshape, const int64_t *yshape, const int32_t xndim, const uint64_t ysize, int& error_code){ */
  /* int threadSize = 256; */
  /* int blockSize = getGridSize(ysize, threadSize);//(ysize + threadSize - 1) / threadSize; */
  /* int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM]; */
  /* int32_t dev_pad[MAX_DIM]; */
  /* get_cuda_shape(xshape, xndim, dev_xshape); */
  /* get_cuda_shape(yshape, xndim, dev_yshape); */
  /* get_cuda_shape(pad_data, xndim, dev_pad); */

  /* kernel_pad<<<blockSize, threadSize>>>(x_data, y_data, xndim, ysize, */
      /* dev_xshape[0], dev_xshape[1], dev_xshape[2], dev_xshape[3], dev_xshape[4], dev_xshape[5], */
      /* dev_yshape[0], dev_yshape[1], dev_yshape[2], dev_yshape[3], dev_yshape[4], dev_yshape[5], */
      /* dev_pad[0], dev_pad[1], dev_pad[2], dev_pad[3], dev_pad[4], dev_pad[5], */
      /* pad_value); */
  /* if(cudaSuccess != cudaGetLastError()){ */
    /* error_code = ERROR_KERNEL; */
  /* } */

  /* return check_cuda_error(cudaGetLastError()); */
/* } */

const char *cuda_expand_dims(const int32_t *ishape_data, int32_t *oshape_data, const int32_t axis, const uint64_t n, int& error_code){
  if(oshape_data == ishape_data){
    return NULL;
  }
  cudaError_t status = cudaMemcpy(oshape_data, ishape_data, sizeof(int32_t) * n, cudaMemcpyDeviceToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
  }
  return check_cuda_error(status);
}

const char *cuda_squeeze(const int32_t *ishape_data, int32_t *oshape_data, const uint64_t n, int& error_code){
  if(oshape_data == ishape_data){
    return NULL;
  }
  cudaError_t status = cudaMemcpy(oshape_data, ishape_data, sizeof(int32_t) * n, cudaMemcpyDeviceToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
  }
  return check_cuda_error(status);
}

__global__ void kernel_transpose(const int32_t *x_data, int32_t *y_data, 
    const int32_t ndim, const int64_t ysize, 
    const int32_t axes_ndim,
    const int64_t xshp0, const int64_t xshp1, const int64_t xshp2, const int64_t xshp3, const int64_t xshp4, const int64_t xshp5,
    const int64_t yshp0, const int64_t yshp1, const int64_t yshp2, const int64_t yshp3, const int64_t yshp4, const int64_t yshp5,
    const int64_t axes0, const int64_t axes1, const int64_t axes2, const int64_t axes3, const int64_t axes4, const int64_t axes5){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int64_t xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int64_t yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  const int64_t axes_data[MAX_DIM] = {axes0, axes1, axes2, axes3, axes4, axes5};
  for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){
    uint64_t in_i = 0, o_i = i;
    for(int j = ndim-1; j >= 0; j--){
      uint64_t col = o_i % yshape[j + MAX_DIM - ndim];
      o_i /= yshape[j + MAX_DIM - ndim];
      int xj = j;
      if(axes_ndim > 0){
        xj = axes_data[j + MAX_DIM - axes_ndim];
      }else{
        xj = ndim - 1 - j;
      }
      int xi = 1;
      for(int tx = ndim-1; tx > xj; tx--){
        xi *= xshape[tx + MAX_DIM - ndim];
      }
      in_i += col * xi;
    }
    y_data[i] = x_data[in_i];
  }
}
const char* cuda_transpose(const int32_t *x_data, const int64_t *axes_data, int32_t *y_data, 
    const int64_t *xshape, const int64_t *yshape, const int32_t ndim, const uint64_t ysize,
    const int32_t axes_ndim, int& error_code){
  int threadSize = 256;
  int blockSize = getGridSize(ysize, threadSize);//(ysize + threadSize - 1) / threadSize;
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM], dev_axes[MAX_DIM];
  get_cuda_shape(xshape, ndim, dev_xshape);
  get_cuda_shape(yshape, ndim, dev_yshape);
  if(axes_ndim > 0){
    get_cuda_shape(axes_data, axes_ndim, dev_axes);
  }

  kernel_transpose<<<blockSize, threadSize>>>(x_data, y_data, ndim, ysize, axes_ndim,
      dev_xshape[0], dev_xshape[1], dev_xshape[2], dev_xshape[3], dev_xshape[4], dev_xshape[5],
      dev_yshape[0], dev_yshape[1], dev_yshape[2], dev_yshape[3], dev_yshape[4], dev_yshape[5],
      dev_axes[0], dev_axes[1], dev_axes[2], dev_axes[3], dev_axes[4], dev_axes[5]);
  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
  }

  return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_stride_slice(const int32_t *x_data, int32_t *y_data,
    const int32_t ndim, const uint32_t ysize,
    const int32_t xshp0, const int32_t xshp1, const int32_t xshp2, const int32_t xshp3, const int32_t xshp4, const int32_t xshp5,
    const int32_t yshp0, const int32_t yshp1, const int32_t yshp2, const int32_t yshp3, const int32_t yshp4, const int32_t yshp5,
    const int32_t bgshp0, const int32_t bgshp1, const int32_t bgshp2, const int32_t bgshp3, const int32_t bgshp4, const int32_t bgshp5,
    const int32_t stshp0, const int32_t stshp1, const int32_t stshp2, const int32_t stshp3, const int32_t stshp4, const int32_t stshp5){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int32_t yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  const int32_t begin_data[MAX_DIM] = {bgshp0, bgshp1, bgshp2, bgshp3, bgshp4, bgshp5};
  const int32_t step_data[MAX_DIM] = {stshp0, stshp1, stshp2, stshp3, stshp4, stshp5};
  const int dim_offset = MAX_DIM - ndim;
  for(uint32_t i = tid; i < ysize; i += gridDim.x*blockDim.x){
    uint32_t o_i = i, in_i = 0, shapeSize = 1;
    for(int j = ndim-1; j >= 0; j--){
      uint32_t col = o_i % yshape[j + dim_offset];
      o_i /= yshape[j + dim_offset];
      int32_t tbegin = begin_data[j + dim_offset];
      int32_t tstep = step_data[j + dim_offset];
      col = tbegin + col * tstep;
      in_i += col * shapeSize;
      shapeSize = shapeSize * xshape[j + dim_offset];
    }
    y_data[i] = x_data[in_i];
  }
}
const char* cuda_stride_slice(const int32_t *x_data, int32_t *y_data, const int64_t *begin_data,
    const int32_t begin_ndim, const int64_t *step_data, const int64_t *xshape, const int64_t *yshape, 
    const int32_t step_ndim, const int32_t y_ndim, const uint64_t ysize, const int32_t x_ndim, int& error_code){
  int threadSize = 256;
  int blockSize = getGridSize(ysize, threadSize);
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM], dev_begin[MAX_DIM], dev_step[MAX_DIM];
  get_cuda_shape(xshape, x_ndim, dev_xshape);
  get_cuda_shape(yshape, y_ndim, dev_yshape);
  get_cuda_shape(begin_data, y_ndim, dev_begin);
  get_cuda_shape(step_data, y_ndim, dev_step);

  kernel_stride_slice<<<blockSize, threadSize>>>(x_data,  y_data, x_ndim, ysize,
      dev_xshape[0], dev_xshape[1], dev_xshape[2], dev_xshape[3], dev_xshape[4], dev_xshape[5],
      dev_yshape[0], dev_yshape[1], dev_yshape[2], dev_yshape[3], dev_yshape[4], dev_yshape[5],
      dev_begin[0], dev_begin[1], dev_begin[2], dev_begin[3], dev_begin[4], dev_begin[5],
      dev_step[0], dev_step[1], dev_step[2], dev_step[3], dev_step[4], dev_step[5]);
  return "";
}

__global__ void kernel_slice_like(const int32_t *x_data, int32_t *y_data, const uint64_t ysize, const int32_t ndim,
    const int64_t xshp0, const int64_t xshp1, const int64_t xshp2, const int64_t xshp3, const int64_t xshp4, const int64_t xshp5,
    const int64_t yshp0, const int64_t yshp1, const int64_t yshp2, const int64_t yshp3, const int64_t yshp4, const int64_t yshp5){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int64_t yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){
    uint64_t o_i = i, in_i = 0, shapeSize = 1;
    for(int j = ndim-1; j >= 0; j--){
      int col = o_i % yshape[j + MAX_DIM - ndim];
      o_i /= yshape[j + MAX_DIM - ndim];
      in_i +=  col * shapeSize;
      shapeSize = shapeSize * xshape[j + MAX_DIM - ndim];
    }
    y_data[i] = x_data[in_i];
  }
}
const char* cuda_slice_like(const int32_t *x_data, int32_t *y_data, const int64_t *xshape, const int64_t *yshape,
    const uint64_t ysize, const int32_t ndim, int& error_code){
  int threadSize = 256;
  int blockSize = getGridSize(ysize, threadSize);//(ysize + threadSize - 1) / threadSize;
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM];
  get_cuda_shape(xshape, ndim, dev_xshape);
  get_cuda_shape(yshape, ndim, dev_yshape);

  kernel_slice_like<<<blockSize, threadSize>>>(x_data, y_data, ysize, ndim,
      dev_xshape[0], dev_xshape[1], dev_xshape[2], dev_xshape[3], dev_xshape[4], dev_xshape[5],
      dev_yshape[0], dev_yshape[1], dev_yshape[2], dev_yshape[3], dev_yshape[4], dev_yshape[5]);
  return "";
}

__global__ void kernel_take(const int32_t *x_data, const int32_t *indices_data, int32_t *y_data, const int32_t yndim,
    const int32_t xndim, const int32_t indices_ndim, const uint64_t ysize, const int32_t axis,
    const int64_t xshp0, const int64_t xshp1, const int64_t xshp2, const int64_t xshp3, const int64_t xshp4, const int64_t xshp5,
    const int64_t yshp0, const int64_t yshp1, const int64_t yshp2, const int64_t yshp3, const int64_t yshp4, const int64_t yshp5,
    const int64_t idshp0, const int64_t idshp1, const int64_t idshp2, const int64_t idshp3, const int64_t idshp4, const int64_t idshp5){
  const int64_t xshape[MAX_DIM] = {xshp0, xshp1, xshp2, xshp3, xshp4, xshp5};
  const int64_t yshape[MAX_DIM] = {yshp0, yshp1, yshp2, yshp3, yshp4, yshp5};
  const int64_t indices_shape[MAX_DIM] = {idshp0, idshp1, idshp2, idshp3, idshp4, idshp5};
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(uint64_t i = tid; i < ysize; i += gridDim.x*blockDim.x){
    uint64_t o_i = i, x_i = 0, indices_i = 0, x_shape_size = 1, indices_shape_size = 1;
    for(int32_t j = yndim - 1, k = indices_ndim-1; j>=axis; j--){
      uint64_t col = o_i % yshape[j + MAX_DIM - yndim];
      o_i /= yshape[j + MAX_DIM - yndim];
      if(j < axis + indices_ndim){
        indices_i += col * indices_shape_size;
        indices_shape_size = indices_shape_size * indices_shape[k + MAX_DIM - indices_ndim];
        --k;
      }
    }

    o_i = i;
    int32_t k = xndim - 1;
    for(int32_t j = yndim - 1; j >= axis + indices_ndim; j--, k--){
      uint64_t col = o_i % yshape[j + MAX_DIM - yndim];
      o_i /= yshape[j + MAX_DIM - yndim];
      x_i += col * x_shape_size;
      x_shape_size = x_shape_size * xshape[k + MAX_DIM - xndim];
    }

    uint64_t x_indices_i = min(max(indices_data[indices_i], 0), (int32_t)xshape[k + MAX_DIM - xndim]-1);
    x_i += x_indices_i * x_shape_size;
    x_shape_size = x_shape_size * xshape[k + MAX_DIM - xndim];
    --k;

    o_i = i;
    for(int32_t j = yndim - 1; j>=0 && k >= 0; j--){
      uint64_t col = o_i % yshape[j + MAX_DIM - yndim];
      o_i /= yshape[j + MAX_DIM - yndim];
      if(j < axis){
        x_i += col * x_shape_size;
        x_shape_size = x_shape_size * xshape[k + MAX_DIM - xndim];
        --k;
      }
    }
    y_data[i] = x_data[x_i];
  }
}
const char* cuda_take(const int32_t *x_data, const int32_t *indices_data, int32_t *y_data, 
    const int64_t *xshape, const int64_t *yshape, const int64_t *indices_shape, const int32_t yndim,
    const int32_t xndim, const int32_t indices_ndim, const uint64_t ysize, const int32_t axis, int& error_code){
  int threadSize = 256;
  int blockSize = getGridSize(ysize, threadSize);//(ysize + threadSize - 1) / threadSize;
  int64_t dev_xshape[MAX_DIM], dev_yshape[MAX_DIM], dev_indices_shape[MAX_DIM];
  get_cuda_shape(xshape, xndim, dev_xshape);
  get_cuda_shape(yshape, yndim, dev_yshape);
  get_cuda_shape(indices_shape, indices_ndim, dev_indices_shape);

  kernel_take<<<blockSize, threadSize>>>(x_data, indices_data, y_data,
      yndim, xndim, indices_ndim, ysize, axis,
      dev_xshape[0], dev_xshape[1], dev_xshape[2], dev_xshape[3], dev_xshape[4], dev_xshape[5],
      dev_yshape[0], dev_yshape[1], dev_yshape[2], dev_yshape[3], dev_yshape[4], dev_yshape[5],
      dev_indices_shape[0], dev_indices_shape[1], dev_indices_shape[2], dev_indices_shape[3], dev_indices_shape[4], dev_indices_shape[5]);

  return "";
}

__global__ void kernel_take_noaxis(const int32_t *x_data, const int32_t *indices_data, int32_t *y_data, const uint64_t ysize, const uint64_t xsize){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){
    int32_t in_i = min((uint64_t)max(indices_data[i], 0), xsize-1); 
    y_data[i] = x_data[in_i];
  }
}
const char* cuda_take(const int32_t *x_data, const int32_t *indices_data, int32_t *y_data, const uint64_t ysize, const uint64_t xsize, int& error_code){
  int threadSize = 256;
  int blockSize = getGridSize(ysize, threadSize);//(ysize + threadSize - 1) / threadSize;
  kernel_take_noaxis<<<blockSize, threadSize>>>(x_data, indices_data, y_data, ysize, xsize);
  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_where_same_shape(const int32_t *x_data, const int32_t *y_data, const int32_t *condition_data, int32_t *result, const uint64_t n){
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(uint64_t i = tid; i < n; i+= gridDim.x * blockDim.x){
    result[i] = condition_data[i] == 0 ? y_data[i] : x_data[i];
  }
}
__global__ void kernel_where_shape0(const int32_t *x_data, const int32_t *y_data, const int32_t *condition_data, int32_t *result, const uint64_t shape0, const uint64_t n){
  int32_t bid = blockIdx.x;
  int32_t tid = threadIdx.x;
  for(int32_t i = bid; i < shape0; i += gridDim.x  * blockDim.x){
    for(int32_t j = tid; j < n; j += blockDim.x){
      result[i * n + j] = (condition_data[i] == 0 ? y_data[i * n + j] : x_data[i * n + j]);
    }
  }
}
const char* cuda_where(const int32_t *x_data, const int32_t *y_data, const int32_t *condition_data, int32_t *result_data, bool same_shape, uint64_t n, uint64_t shape0, int& error_code){
  if(same_shape){
    const int32_t threadSize = 256;
    const int32_t blockSize = getGridSize(n, threadSize);
    kernel_where_same_shape<<<blockSize, threadSize>>>(x_data, y_data, condition_data, result_data, n);
  }else{
    const int32_t threadSize = 256;
    const int32_t blockSize = shape0;
    kernel_where_shape0<<<blockSize, threadSize>>>(x_data, y_data, condition_data, result_data, shape0, n);
  }
  cudaError_t error = cudaGetLastError();
  return check_cuda_error(error);
}
}
}
