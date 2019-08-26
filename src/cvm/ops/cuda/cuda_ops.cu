#include "cuda_ops.h"
#include "../common.h"

#define BS 16
#define FS 8
__global__ void kernel_conv2d(
    const int32_t * __restrict__ input, const int32_t i_n, const int32_t i_c, const int32_t i_h, const int32_t i_w,
    const int32_t * __restrict__ filter, const int32_t f_n, const int32_t f_c, const int32_t f_h, const int32_t f_w,
    const int32_t * __restrict__ bias,
    const int32_t padding_h, const int32_t padding_w,
    const int32_t stride_h, const int32_t stride_w,
    const int32_t dilation_h, const int32_t dilation_w,
    const int32_t groups,
    int32_t *output, const int32_t o_n, const int32_t o_c, const int32_t o_h, const int32_t o_w){
  int g_x = blockDim.x * blockIdx.x + threadIdx.x;
  int l_y = threadIdx.y; 
  int l_x = threadIdx.x;
  int tmp_f_h = (f_h - 1) * dilation_h + 1; // for dilation, to be optimized
  int tmp_f_w = (f_w - 1) * dilation_w + 1;
  int tmp_o_h = i_h + 2 * padding_h - tmp_f_h + 1; // for stride
  int tmp_o_w = i_w + 2 * padding_w - tmp_f_w + 1;
  int perBlockOneImageY = (tmp_o_h+BS-1) / BS;
  int perBlockOneImageX = (tmp_o_w+BS-1) / BS;
  int l_o_c = blockIdx.y / perBlockOneImageY;
  int n = l_o_c / ((o_c+FS-1)/FS);
  int nsize = n * i_c * i_h * i_w; 
  int l_f_n = l_o_c % ((o_c+FS-1)/FS);
  int l_o_hi = blockIdx.y % perBlockOneImageY;
  int l_o_wi = blockIdx.x % perBlockOneImageX;
  int l_o_h = l_o_hi * BS + l_y;
  //    int l_o_w = l_o_wi * BS + l_x;

  const int32_t F_H = f_h;
  const int32_t F_W = f_w;
  //    __shared__ int32_t shared_i[BS + F_H - 1][BS + F_W - 1];
  int32_t sih = BS + tmp_f_h - 1;
  int32_t siw = BS + tmp_f_w - 1;
  extern __shared__ int32_t  share[];
  int32_t *shared_i = (int32_t*)share; 
  int32_t *shared_f = &share[sih * siw];
  int32_t *shared_b = &shared_f[F_H*F_W*FS];

  int32_t sum[FS] = {0}; 
  int min_s_y = (l_o_hi+1) * BS <= tmp_o_h ? BS : tmp_o_h%BS;
  int min_s_x = (l_o_wi+1) * BS <= tmp_o_w ? BS : tmp_o_w%BS;

  //load bias to shared memory
  int lid = l_y * BS + l_x;
  for(int i = lid; bias != NULL && i < FS; i+=BS*BS){
    if(l_f_n*FS + i < o_c)
      shared_b[i] = bias[l_f_n*FS + i];
    else shared_b[i] = 0;
  }

  if(l_o_h >= tmp_o_h || g_x >= tmp_o_w) return;

  for(int c = 0; c < i_c; c++){
    //load input to shared
    int l_i_h = l_o_h - padding_h;
    int i_y = c * i_h + l_i_h;
    int i_x = g_x - padding_w;
    if(l_i_h < 0 || i_x < 0 || l_i_h >= i_h || i_x >= i_w)
      shared_i[l_y*siw + l_x] = 0;
    else
      shared_i[l_y*siw + l_x] = input[nsize + i_y * i_w + i_x];

    if(l_y < tmp_f_h-1){
      for(int i = l_y; i < tmp_f_h-1; i+=min_s_y){
        if(l_i_h+min_s_y+i-l_y < 0 || i_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x >= i_w)
          shared_i[(i+min_s_y)*siw + l_x] = 0;
        else
          shared_i[(i + min_s_y)*siw + l_x] = input[nsize + (i_y + min_s_y + i - l_y) * i_w + i_x];     
      }
    }
    if(l_x < tmp_f_w-1){
      for(int i = l_x; i < tmp_f_w-1; i+= min_s_x){
        if(l_i_h < 0 || i_x+min_s_x+i-l_x < 0 || l_i_h >= i_h || i_x+min_s_x+i-l_x >= i_w)
          shared_i[l_y * siw + i+min_s_x] = 0;
        else
          shared_i[l_y * siw + i + min_s_x] = input[nsize + i_y * i_w + i_x + min_s_x + i - l_x];
      }
    }
    if(l_y < tmp_f_h-1 && l_x < tmp_f_w-1){
      for(int i = l_y; i < tmp_f_h-1; i+=min_s_y){
        for(int j = l_x; j < tmp_f_w-1; j+=min_s_x){
          if(l_i_h+min_s_y+i-l_y < 0 || i_x+min_s_x+j-l_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x+min_s_x+j-l_x >= i_w)
            shared_i[(i+min_s_y) * siw + j+min_s_x] = 0;
          else
            shared_i[(i+min_s_y) * siw + j+min_s_x] = input[nsize + (i_y+min_s_y + i-l_y)*i_w + i_x + min_s_x + j - l_x];
        }
      }
    }

    //load filter to shared;
    if(l_y < F_H && l_x < F_W){
      for(int i = l_y; i < F_H; i+= min_s_y)
        for(int j = l_x; j < F_W; j+=min_s_x)
          for(int fc = 0; fc < FS; fc++){
            if(l_f_n * FS + fc < o_c)
              shared_f[fc * F_H*F_W + i*F_W + j] = filter[(l_f_n*FS+fc) * F_H * F_W * f_c + c * F_H * F_W + i * F_W + j];
            else shared_f[fc * F_H * F_W + i * F_W + j] = 0;
          }
    }
    __syncthreads();

    for(int fy = 0; fy < F_H; fy++){
      for(int fx = 0; fx < F_W; fx++){
        int32_t tmpx = shared_i[(l_y+fy*dilation_h)*siw + l_x+fx*dilation_w];
#pragma unroll
        for(int fc = 0; fc < FS; fc++){
          sum[fc] += tmpx * shared_f[fc*F_H*F_W + fy*F_W + fx];
        }
      }
    } 
    __syncthreads();
  }

  if(l_o_h % stride_h == 0 && g_x % stride_w == 0){ //TODO to be optimized
    //int oi = l_o_c * o_h * o_w + l_o_h * o_w + g_x;
    for(int fc = 0; fc < FS; fc++){
      if(l_f_n*FS + fc < o_c){
        int oi = n*o_c*o_h*o_w + (l_f_n*FS+fc) * o_h * o_w + l_o_h/stride_h * o_w + g_x/stride_w;
        output[oi] = sum[fc] + (bias != NULL ? shared_b[fc] : 0);
      }
    }
  }
}

__global__ void kernel_conv2d_no_shared(
    const int32_t * __restrict__ input, const int32_t i_n, const int32_t i_c, const int32_t i_h, const int32_t i_w,
    const int32_t * __restrict__ filter, const int32_t f_n, const int32_t f_c, const int32_t f_h, const int32_t f_w,
    const int32_t * __restrict__ bias,
    const int32_t padding_h, const int32_t padding_w,
    const int32_t stride_h, const int32_t stride_w,
    const int32_t dilation_h, const int32_t dilation_w,
    const int32_t groups,
    int32_t *output, const int32_t o_n, const int32_t o_c, const int32_t o_h, const int32_t o_w){
  int32_t gy = threadIdx.y + blockIdx.y * blockDim.y;
  int32_t gx = threadIdx.x + blockIdx.x * blockDim.x;
  int32_t l_o_h = gy % o_h;
  int32_t l_o_c = gy / o_h % o_c;
  int32_t l_o_n = gy / (o_h * o_c);
  if(gy < o_n * o_c * o_h && gx < o_w){
    int32_t sum = 0;
    for(int ic = 0; ic < i_c; ++ic){
      for(int fy = 0; fy < f_h; ++fy){
        for(int fx = 0; fx < f_w; ++fx){
          int32_t l_i_h = l_o_h * stride_h + fy * dilation_h - padding_h;
          int32_t l_i_w = gx * stride_w + fx * dilation_h - padding_w;
          int32_t x;
          if(l_i_h < 0 || l_i_w < 0 || l_i_h >= i_h || l_i_w >= i_w)
            x = 0;
          else x = input[l_o_n * i_c * i_h * i_w + ic * i_h * i_w + l_i_h * i_w + l_i_w];
          sum += x * filter[l_o_c * i_c * f_h * f_w + ic * f_h * f_w + fy * f_w + fx];
        }
      }
    }
    output[gy * o_w + gx] = sum + (bias != NULL ? bias[l_o_c] : 0);
  }
}
__global__ void kernel_int32_to_int8(const int32_t *in_data, int8_t *out_data, const int n){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for(int64_t i = tid; i < n; i+= gridDim.x * blockDim.x){
        out_data[i] = static_cast<int8_t>(in_data[i]);
    }
}

__global__ void im2col_gpu_kernel(const int n, const int32_t* data_im,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int height_col, const int width_col,
        int8_t* data_col) {
//    CUDA_KERNEL_LOOP(index, n) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int64_t index = tid; index < n; index += gridDim.x*blockDim.x){
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;
        int8_t* data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const int32_t* data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                *data_col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                    static_cast<int8_t>(data_im_ptr[i * dilation_h * width + j * dilation_w]) : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

#define TILE_WIDTH 16
__global__ void kernel_matrix_mul(
    int8_t *a, // m*k 
    int8_t *b, // k*n
    int32_t *c, // m*n
    int32_t m, int32_t k, int32_t n, int32_t *bias){
  __shared__ int8_t sharedm[TILE_WIDTH][TILE_WIDTH];
  __shared__ int8_t sharedn[TILE_WIDTH][TILE_WIDTH];
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
      sharedm[ty][tx] = a[row*k + i*TILE_WIDTH + tx];
    else
      sharedm[ty][tx] = 0;

    if(i*TILE_WIDTH + ty < k && col < n)//k*n
      sharedn[ty][tx] =b[(i*TILE_WIDTH + ty) * n + col] ;//b[col * k + i * TILE_WIDTH + ty];
    else
      sharedn[ty][tx] = 0;
    __syncthreads();

    for(int j = 0; j < TILE_WIDTH; j++)
      sum += static_cast<int32_t>(sharedm[ty][j]) * sharedn[j][tx];
    __syncthreads();
  }
  if (row < m && col < n){
    if(bias != NULL) sum += bias[row];
    c[row*n + col] = sum;
  }
}
inline void im2col_gpu(const int32_t* data_im, const int channels,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        int8_t* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h -
            (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w -
            (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    int threads = 256;
    int blocks = (num_kernels + threads - 1) / threads;
    im2col_gpu_kernel<<<blocks, threads>>>(
                num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
                pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
                width_col, data_col);
}
const char* cuda_conv2d(
    int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
    int32_t *filter, int32_t f_n, int32_t f_c, const int32_t f_h, const int32_t f_w,
    int32_t *bias,
    const int32_t padding_h, const int32_t padding_w,
    const int32_t stride_h, const int32_t stride_w,
    const int32_t dilation_h, const int32_t dilation_w,
    int32_t groups,
    int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, 
    int32_t device_id,
    int& error_code){
  if(i_n < 1 || i_c < 1 || i_h < 1 || i_w < 1 || f_n < 1 || f_c < 1 || f_h < 1 || f_w < 1 || 
      padding_h < 0 || padding_w < 0 || stride_h < 1 || stride_w < 1 || dilation_h < 1 || dilation_w < 1 ||
      o_n < 1 || o_c < 1 || o_h < 1 || o_w < 1){
    error_code = ERROR_PARAMS;
    return "error args";
  }
  int32_t *dev_i = input, *dev_f = filter, *dev_o = output, *dev_b = bias;

  int tmp_f_h = (f_h - 1) * dilation_h + 1; // for dilation, to be optimized
  int tmp_f_w = (f_w - 1) * dilation_w + 1;
  int tmp_o_h = i_h + 2 * padding_h - tmp_f_h + 1; //for stride > 1 , TODO to be optimized
  int tmp_o_w = i_w + 2 * padding_w - tmp_f_w + 1;
  int32_t totalShareMemSize = getShareMemorySize(device_id, error_code);
  if(error_code != NON_ERROR){
    return check_cuda_error(cudaGetLastError());
  }
  size_t share_size = ((BS + tmp_f_h - 1) * (BS + tmp_f_w - 1) + f_h * f_w * FS + FS) * sizeof(int32_t);
  if(share_size < totalShareMemSize){
    size_t freeSize = getFreeMemorySize(device_id, error_code);
    size_t tmp_filter_size = o_c * i_c * f_h * f_w * sizeof(int8_t);
    size_t tmp_input_size = i_c * f_h * f_w * o_h * o_w * sizeof(int8_t);
    if(tmp_filter_size + tmp_input_size >= freeSize || tmp_filter_size > MEMORY_LIMIT || tmp_input_size > MEMORY_LIMIT){
      int b_h = BS;
      int b_w = BS;
      int32_t g_h = o_n * ((o_c + FS - 1) / FS) * ((tmp_o_h + b_h - 1) / b_h);
      int32_t g_w = (tmp_o_w + b_w - 1) / b_w;
      dim3 bDim(b_w, b_h, 1);
      dim3 gDim(g_w, g_h, 1);
      kernel_conv2d<<<gDim, bDim, share_size>>>(
          dev_i, i_n, i_c, i_h, i_w,
          dev_f, f_n, f_c, f_h, f_w,
          dev_b, 
          padding_h, padding_w,
          stride_h, stride_w,
          dilation_h, dilation_w,
          groups,
          dev_o, o_n, o_c, o_h, o_w);
    }else{
      int32_t fn = o_c * i_c * f_h * f_w;
      const int M = o_c;
      const int K = i_c * f_h * f_w;
      const int N = o_h * o_w;
      dim3 bDim(TILE_WIDTH, TILE_WIDTH, 1);
      int gh = (M + TILE_WIDTH - 1) / TILE_WIDTH;
      int gw = (N + TILE_WIDTH - 1) / TILE_WIDTH;
      dim3 gDim(gw, gh, 1);

      int8_t *d_f, *d_col;
      cudaError_t status = cudaMalloc((void**)&d_f, fn * sizeof(int8_t));
      if(status != cudaSuccess){
        error_code = ERROR_MALLOC;
        return check_cuda_error(status);
      }
      int blockSize = 256;
      int gridSize = getGridSize(fn, blockSize);
      kernel_int32_to_int8<<<gridSize, blockSize>>>(dev_f, d_f, fn);

      status = cudaMalloc((void**)&d_col, sizeof(int8_t) * i_c * f_h * f_w * o_h * o_w);
      if(status != cudaSuccess){
        cudaFree(d_f);
        error_code = ERROR_MALLOC;
        return check_cuda_error(status);
      }
      
      for(int i = 0; i < o_n; i++){
        im2col_gpu(dev_i + i * i_c * i_h * i_w,
            i_c, i_h, i_w, f_h, f_w, padding_h, padding_w, stride_h, stride_w, 
            dilation_h, dilation_w, d_col);
        kernel_matrix_mul<<<gDim, bDim>>>(d_f, d_col, dev_o + i * o_c * o_h * o_w, M, K, N, dev_b);
      }
      cudaFree(d_f);
      cudaFree(d_col);
    }
  }else{
    int b_h = BS;
    int b_w = BS;
    int g_h = o_n * o_c * ((o_h + b_h - 1) / b_h);
    int g_w = (o_w + b_w - 1) / b_w;
    dim3 bDim(b_w, b_h, 1);
    dim3 gDim(g_w, g_h, 1);
    kernel_conv2d_no_shared<<<gDim, bDim>>>(
        dev_i, i_n, i_c, i_h, i_w,
        dev_f, f_n, f_c, f_h, f_w,
        dev_b, 
        padding_h, padding_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        groups,
        dev_o, o_n, o_c, o_h, o_w);
  }
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }

  print_to_file(dev_o, o_n * o_c * o_h * o_w, "/tmp/zkh/trec/gpu/conv2d.txt");
  return check_cuda_error(error);
}
__global__ void kernel_depthwise_conv2d(
    const int32_t * __restrict__ input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
    const int32_t * __restrict__ filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
    const int32_t * __restrict__ bias,
    int32_t padding_h, int32_t padding_w,
    int32_t stride_h, int32_t stride_w,
    int32_t dilation_h, int32_t dilation_w, 
    int32_t groups,
    int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w)
{
  int g_x = blockDim.x * blockIdx.x + threadIdx.x;
  int l_y = threadIdx.y; 
  int l_x = threadIdx.x;
  int tmp_f_h = (f_h - 1) * dilation_h + 1; // for dilation, to be optimized
  int tmp_f_w = (f_w - 1) * dilation_w + 1;
  int tmp_o_h = i_h + 2 * padding_h - tmp_f_h + 1; // for stride
  int tmp_o_w = i_w + 2 * padding_w - tmp_f_w + 1;
  int perBlockOneImageY = (tmp_o_h+BS-1) / BS;
  int perBlockOneImageX = (tmp_o_w+BS-1) / BS;
  int l_o_c = blockIdx.y / perBlockOneImageY;
  int l_f_c = l_o_c % o_c;
  int l_o_hi = blockIdx.y % perBlockOneImageY;
  int l_o_wi = blockIdx.x % perBlockOneImageX;
  int l_o_h = l_o_hi * BS + l_y;
  //    int l_o_w = l_o_wi * BS + l_x;
  if(l_o_h >= tmp_o_h || g_x >= tmp_o_w) return;

  const int32_t F_H = f_h;
  const int32_t F_W = f_w;
  //    __shared__ int32_t shared_i[BS + F_H - 1][BS + F_W - 1];
  int32_t sih = BS + tmp_f_h - 1;
  int32_t siw = BS + tmp_f_w - 1;
  extern __shared__ int32_t  share[];
  int32_t *shared_i = (int32_t*)share; 
  int32_t *shared_f = &share[sih * siw];

  int32_t sum = 0; 
  int min_s_y = (l_o_hi+1) * BS <= tmp_o_h ? BS : tmp_o_h%BS;
  int min_s_x = (l_o_wi+1) * BS <= tmp_o_w ? BS : tmp_o_w%BS;

  //load input to shared
  int l_i_h = l_o_h - padding_h;
  int i_y = l_o_c * i_h + l_i_h;
  int i_x = g_x - padding_w;
  // 0~2-> -1~1
  if(l_i_h < 0 || i_x < 0 || l_i_h >= i_h || i_x >= i_w)
    shared_i[l_y*siw + l_x] = 0;
  else
    shared_i[l_y*siw + l_x] = input[i_y * i_w + i_x];

  if(l_y < tmp_f_h-1){
    for(int i = l_y; i < tmp_f_h-1; i+=min_s_y){
      if(l_i_h+min_s_y+i-l_y < 0 || i_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x >= i_w)
        shared_i[(i+min_s_y)*siw + l_x] = 0;
      else
        shared_i[(i + min_s_y)*siw + l_x] = input[(i_y + min_s_y + i - l_y) * i_w + i_x]; 
    }
  }
  if(l_x < tmp_f_w-1){
    for(int i = l_x; i < tmp_f_w-1; i+= min_s_x){
      if(l_i_h < 0 || i_x+min_s_x+i-l_x < 0 || l_i_h >= i_h || i_x+min_s_x+i-l_x >= i_w)
        shared_i[l_y * siw + i+min_s_x] = 0;
      else
        shared_i[l_y * siw + i + min_s_x] = input[i_y * i_w + i_x + min_s_x + i - l_x];
    }
  }
  if(l_y < tmp_f_h-1 && l_x < tmp_f_w-1){
    for(int i = l_y; i < tmp_f_h-1; i+=min_s_y){
      for(int j = l_x; j < tmp_f_w-1; j+=min_s_x){
        if(l_i_h+min_s_y+i-l_y < 0 || i_x+min_s_x+j-l_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x+min_s_x+j-l_x >= i_w)
          shared_i[(i+min_s_y) * siw + j+min_s_x] = 0;
        else
          shared_i[(i+min_s_y) * siw + j+min_s_x] = input[(i_y+min_s_y + i-l_y)*i_w + i_x + min_s_x + j - l_x];
      }
    }
  }

  //load filter to shared;
  if(l_y < F_H && l_x < F_W){
    for(int i = l_y; i < F_H; i+= min_s_y)
      for(int j = l_x; j < F_W; j+=min_s_x)
        shared_f[i*F_W + j] = filter[l_f_c * F_H * F_W + i * F_W + j];
  }
  __syncthreads();

  for(int fy = 0; fy < F_H; fy++){
    for(int fx = 0; fx < F_W; fx++){
      sum += shared_i[(l_y+fy*dilation_h)*siw + l_x+fx*dilation_w] * shared_f[fy*F_W + fx];
    }
  } 
  __syncthreads();

  if(l_o_h % stride_h == 0 && g_x % stride_w == 0){
    //int oi = l_o_c * o_h * o_w + l_o_h * o_w + g_x;
    int oi = l_o_c * o_h * o_w + l_o_h/stride_h * o_w + g_x/stride_w;
    output[oi] = sum + (bias != NULL ? bias[l_o_c%o_c] : 0);
  }
}
__global__ void kernel_depthwise_conv2d_no_shared(
    int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
    int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
    int32_t *bias,
    int32_t padding_h, int32_t padding_w,
    int32_t stride_h, int32_t stride_w,
    int32_t dilation_h, int32_t dilation_w, 
    int32_t groups,
    int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
  int32_t gy = threadIdx.y + blockIdx.y * blockDim.y;
  int32_t gx = threadIdx.x + blockIdx.x * blockDim.x;
  int32_t l_o_h = gy % o_h;
  int32_t l_o_c = gy / o_h % o_c;
  int32_t l_o_n = gy / (o_h * o_c);
  if(gy < o_n * o_c * o_h && gx < o_w){
    int32_t sum = 0;
    for(int fy = 0; fy < f_h; ++fy){
      for(int fx = 0; fx < f_w; ++fx){
        int32_t l_i_h = l_o_h * stride_h + fy * dilation_h - padding_h;
        int32_t l_i_w = gx * stride_w + fx * dilation_w - padding_w;
        int32_t x;
        if(l_i_h < 0 || l_i_w < 0 || l_i_h >= i_h || l_i_w >= i_w)
          //x = 0;
          continue;
        x = input[l_o_n * i_c * i_h * i_w + l_o_c * i_h * i_w + l_i_h * i_w + l_i_w];
        sum += x * filter[l_o_c * f_h * f_w + fy * f_w + fx];
      }
    }
    output[gy * o_w + gx] = sum + (bias != NULL ? bias[l_o_c] : 0);
  }
}
const char* cuda_depthwise_conv2d(
    int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
    int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
    int32_t *bias,
    int32_t padding_h, int32_t padding_w,
    int32_t stride_h, int32_t stride_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t groups,
    int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, int& error_code){
  int32_t *dev_i = input, *dev_f = filter, *dev_o = output, *dev_b = bias;

  int b_h = BS;
  int b_w = BS;
  int tmp_f_h = (f_h - 1) * dilation_h + 1; // for dilation, to be optimized
  int tmp_f_w = (f_w - 1) * dilation_w + 1;
  int tmp_o_h = i_h + 2 * padding_h - tmp_f_h + 1; //for stride > 1
  int tmp_o_w = i_w + 2 * padding_w - tmp_f_w + 1;
  const int32_t totalShareMemSize = getShareMemorySize(device_id, error_code);
  if(error_code != NON_ERROR){
    return check_cuda_error(cudaGetLastError());
  }
  size_t share_size = (BS + tmp_f_h - 1) * (BS + tmp_f_w - 1) * sizeof(int32_t) + f_h * f_w * sizeof(int32_t);
  if(false){//(share_size < totalShareMemSize){
    int32_t g_h = o_n * o_c * ((tmp_o_h + b_h - 1) / b_h);
    int32_t g_w = (tmp_o_w + b_w - 1) / b_w;
    dim3 bDim(b_w, b_h, 1);
    dim3 gDim(g_w, g_h, 1);
    kernel_depthwise_conv2d<<<gDim, bDim, share_size>>>(
        dev_i, i_n, i_c, i_h, i_w,
        dev_f, f_n, f_c, f_h, f_w,
        dev_b, 
        padding_h, padding_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        groups,
        dev_o, o_n, o_c, o_h, o_w);
  }else{
    int32_t g_h = o_n * o_c * ((tmp_o_h + b_h - 1) / b_h); 
    int32_t g_w = (tmp_o_w + b_w - 1) / b_w;
    dim3 bDim(b_w, b_h, 1);
    dim3 gDim(g_w, g_h, 1);
    kernel_depthwise_conv2d_no_shared<<<gDim, bDim>>>(
        dev_i, i_n, i_c, i_h, i_w,
        dev_f, f_n, f_c, f_h, f_w,
        dev_b, 
        padding_h, padding_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        groups,
        dev_o, o_n, o_c, o_h, o_w);
  }
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}
__global__ void kernel_groupwise_conv2d_no_shared(
    int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
    int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
    int32_t *bias,
    int32_t padding_h, int32_t padding_w,
    int32_t stride_h, int32_t stride_w,
    int32_t dilation_h, int32_t dilation_w, 
    int32_t groups,
    int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
  int32_t gy = threadIdx.y + blockIdx.y * blockDim.y;
  int32_t gx = threadIdx.x + blockIdx.x * blockDim.x;
  int32_t l_o_h = gy % o_h;
  int32_t l_o_c = gy / o_h % o_c;
  int32_t l_o_n = gy / (o_h * o_c);
  const int32_t ochannels_per_group = o_c / groups;
  const int32_t ichannels_per_group = i_c / groups;
  if(gy < o_n * o_c * o_h && gx < o_w){
    int32_t sum = 0;
    int32_t ic = l_o_c / ochannels_per_group * ichannels_per_group;
    for(int tic = 0; tic < ichannels_per_group; ++tic){
      for(int fy = 0; fy < f_h; ++fy){
        for(int fx = 0; fx < f_w; ++fx){
          int32_t l_i_h = l_o_h * stride_h + fy * dilation_h - padding_h;
          int32_t l_i_w = gx * stride_w + fx * dilation_w - padding_w;
          int32_t x;
          if(l_i_h < 0 || l_i_w < 0 || l_i_h >= i_h || l_i_w >= i_w)
            continue;
          x = input[l_o_n * i_c * i_h * i_w + (ic+tic) * i_h * i_w + l_i_h * i_w + l_i_w];
          sum += x * filter[l_o_c * f_h * f_w * f_c + tic * f_h * f_w + fy * f_w + fx];
        }
      }
    }
    output[gy * o_w + gx] = sum + (bias != NULL ? bias[l_o_c] : 0);
  }
}
const char* cuda_groupwise_conv2d(
    int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
    int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
    int32_t *bias,
    int32_t padding_h, int32_t padding_w,
    int32_t stride_h, int32_t stride_w,
    int32_t dilation_h, int32_t dilation_w,
    int32_t groups,
    int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, int& error_code){
  int32_t *dev_i = input, *dev_f = filter, *dev_o = output, *dev_b = bias;

  int b_h = BS;
  int b_w = BS;
  int tmp_f_h = (f_h - 1) * dilation_h + 1; 
  int tmp_f_w = (f_w - 1) * dilation_w + 1;
  int tmp_o_h = i_h + 2 * padding_h - tmp_f_h + 1; 
  int tmp_o_w = i_w + 2 * padding_w - tmp_f_w + 1;
  int32_t g_h = o_n * o_c * ((tmp_o_h + b_h - 1) / b_h); 
  int32_t g_w = (tmp_o_w + b_w - 1) / b_w;
  dim3 bDim(b_w, b_h, 1);
  dim3 gDim(g_w, g_h, 1);
  kernel_groupwise_conv2d_no_shared<<<gDim, bDim>>>(
      dev_i, i_n, i_c, i_h, i_w,
      dev_f, f_n, f_c, f_h, f_w,
      dev_b, 
      padding_h, padding_w,
      stride_h, stride_w,
      dilation_h, dilation_w,
      groups,
      dev_o, o_n, o_c, o_h, o_w);
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}

__global__ void kernel_max_pool(
    int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
    int32_t f_h, int32_t f_w,
    int32_t padding_h, int32_t padding_w,
    int32_t stride_h, int32_t stride_w,
    int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
  int g_x = blockDim.x * blockIdx.x + threadIdx.x;
  int l_y = threadIdx.y; 
  int l_x = threadIdx.x;
  int tmp_o_h = i_h + 2 * padding_h - f_h + 1; // for stride
  int tmp_o_w = i_w + 2 * padding_w - f_w + 1;
  int perBlockOneImageY = (tmp_o_h+BS-1) / BS;
  int perBlockOneImageX = (tmp_o_w+BS-1) / BS;
  int l_o_c = blockIdx.y / perBlockOneImageY;
  int l_o_hi = blockIdx.y % perBlockOneImageY;
  int l_o_wi = blockIdx.x % perBlockOneImageX;
  int l_o_h = l_o_hi * BS + l_y;
  //    int l_o_w = l_o_wi * BS + l_x;
  if(l_o_h >= tmp_o_h || g_x >= tmp_o_w) return;

  const int32_t F_H = f_h;
  const int32_t F_W = f_w;
  //    __shared__ int32_t shared_i[BS + F_H - 1][BS + F_W - 1];
  //    int32_t sih = BS + F_H - 1;
  int32_t siw = BS + F_W - 1;
  extern __shared__ int32_t  share[];
  int32_t *shared_i = (int32_t*)share; 

  int32_t minV = int32_t(1)<<31; 
  int32_t max_elem = minV;
  int min_s_y = (l_o_hi+1) * BS <= tmp_o_h ? BS : tmp_o_h%BS;
  int min_s_x = (l_o_wi+1) * BS <= tmp_o_w ? BS : tmp_o_w%BS;

  //load input to shared
  int l_i_h = l_o_h - padding_h;
  int i_y = l_o_c * i_h + l_i_h;
  int i_x = g_x - padding_w;
  // 0~2-> -1~1
  if(l_i_h < 0 || i_x < 0 || l_i_h >= i_h || i_x >= i_w)
    shared_i[l_y*siw + l_x] = minV;
  else
    shared_i[l_y*siw + l_x] = input[i_y * i_w + i_x];

  if(l_y < F_H-1){
    for(int i = l_y; i < F_H-1; i+=min_s_y){
      if(l_i_h+min_s_y+i-l_y < 0 || i_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x >= i_w)
        shared_i[(i+min_s_y)*siw + l_x] = minV;
      else
        shared_i[(i + min_s_y)*siw + l_x] = input[(i_y + min_s_y + i - l_y) * i_w + i_x];     
    }
  }
  if(l_x < F_W-1){
    for(int i = l_x; i < F_W-1; i+= min_s_x){
      if(l_i_h < 0 || i_x+min_s_x+i-l_x < 0 || l_i_h >= i_h || i_x+min_s_x+i-l_x >= i_w)
        shared_i[l_y * siw + i+min_s_x] = minV;
      else
        shared_i[l_y * siw + i + min_s_x] = input[i_y * i_w + i_x + min_s_x + i - l_x];
    }
  }
  if(l_y < F_H-1 && l_x < F_W-1){
    for(int i = l_y; i < F_H-1; i+=min_s_y){
      for(int j = l_x; j < F_W-1; j+=min_s_x){
        if(l_i_h+min_s_y+i-l_y < 0 || i_x+min_s_x+j-l_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x+min_s_x+j-l_x >= i_w)
          shared_i[(i+min_s_y) * siw + j+min_s_x] = minV;
        else
          shared_i[(i+min_s_y) * siw + j+min_s_x] = input[(i_y+min_s_y + i-l_y)*i_w + i_x + min_s_x + j - l_x];
      }
    }
  }
  __syncthreads();

  for(int fy = 0; fy < F_H; fy++){
    for(int fx = 0; fx < F_W; fx++){
      int32_t tmp =  shared_i[(l_y+fy)*siw + l_x+fx];
      max_elem = max_elem < tmp ? tmp : max_elem;
    }
  } 
  __syncthreads();

  if(l_o_h % stride_h == 0 && g_x % stride_w == 0){
    //int oi = l_o_c * o_h * o_w + l_o_h * o_w + g_x;
    int oi = l_o_c * o_h * o_w + l_o_h/stride_h * o_w + g_x/stride_w;
    output[oi] = max_elem;
  }
}

__global__ void kernel_max_pool_no_shared(
    int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
    int32_t f_h, int32_t f_w,
    int32_t padding_h, int32_t padding_w,
    int32_t stride_h, int32_t stride_w,
    int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
  int32_t gy = threadIdx.y + blockIdx.y * blockDim.y;
  int32_t gx = threadIdx.x + blockIdx.x * blockDim.x;
  int32_t l_o_h = gy % o_h;
  int32_t l_o_c = gy / o_h % o_c;
  int32_t l_o_n = gy / (o_h * o_c);
  if(gy < o_n * o_c * o_h && gx < o_w){
    int32_t minV = (int32_t)1 << 31;
    int32_t maxV = minV;
    for(int fy = 0; fy < f_h; ++fy){
      for(int fx = 0; fx < f_w; ++fx){
        int32_t l_i_h = l_o_h * stride_h + fy  - padding_h;
        int32_t l_i_w = gx * stride_w + fx - padding_w;
        int32_t x;
        if(l_i_h < 0 || l_i_w < 0 || l_i_h >= i_h || l_i_w >= i_w)
          x = minV;
        else x = input[l_o_n * i_c * i_h * i_w + l_o_c * i_h * i_w + l_i_h * i_w + l_i_w];
        maxV = maxV < x ? x : maxV;
      }
    }
    output[gy * o_w + gx] = maxV;
  }
}
const char* cuda_max_pool(
    int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
    const int32_t f_h, const int32_t f_w,
    int32_t padding_h, int32_t padding_w,
    int32_t stride_h, int32_t stride_w,
    int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, int& error_code){
  int32_t *dev_i = input, *dev_o = output;

  const int32_t totalShareMemSize = getShareMemorySize(device_id, error_code);
  if(error_code != NON_ERROR){
    return check_cuda_error(cudaGetLastError());
  }
  size_t share_size = (BS + f_h - 1) * (BS + f_w - 1) * sizeof(int32_t);
  int b_h = BS;
  int b_w = BS;
  int tmp_o_h = i_h + 2 * padding_h - f_h + 1; //for stride > 1
  int tmp_o_w = i_w + 2 * padding_w - f_w + 1;
  if(false){//(share_size < totalShareMemSize){
    int32_t g_h = o_n * o_c * ((tmp_o_h + b_h - 1) / b_h);
    int32_t g_w = (tmp_o_w + b_w - 1) / b_w;
    dim3 bDim(b_w, b_h, 1);
    dim3 gDim(g_w, g_h, 1);
    kernel_max_pool<<<gDim, bDim, share_size>>>(
        dev_i, i_n, i_c, i_h, i_w,
        f_h, f_w,
        padding_h, padding_w, 
        stride_h, stride_w,
        dev_o, o_n, o_c, o_h, o_w);
  }else{
    int32_t g_h = o_n * o_c * ((o_h + b_h - 1) / b_h); 
    int32_t g_w = (o_w + b_w - 1) / b_w;
    dim3 bDim(b_w, b_h, 1);
    dim3 gDim(g_w, g_h, 1);
    kernel_max_pool_no_shared<<<gDim, bDim>>>(
        dev_i, i_n, i_c, i_h, i_w,
        f_h, f_w,
        padding_h, padding_w, 
        stride_h, stride_w,
        dev_o, o_n, o_c, o_h, o_w);
  }
  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  return check_cuda_error(error);
}

//#define TILE_WIDTH 16
__global__ void kernel_dense(
    int32_t *A, // m*k 
    int32_t *B, // was transposed, n*k
    int32_t *C, // m*n
    int32_t m, int32_t k, int32_t n, int32_t *bias, int32_t useBias){
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

    if(i*TILE_WIDTH + ty < k && col < n)//n*k
      sharedN[tx][ty] = B[col * k + i * TILE_WIDTH + ty];
    else
      sharedN[tx][ty] = 0;
    __syncthreads();

    for(int j = 0; j < TILE_WIDTH; j++)
      sum += sharedM[ty][j] * sharedN[tx][j];
    __syncthreads();
  }
  if (row < m && col < n){
    if(useBias == 1) sum += bias[col];
    C[row*n + col] = sum;
  }
}

const char* cuda_dense(
    int32_t *a,
    int32_t *b,
    int32_t *c,
    const int m, const int k, const int n, int32_t* bias, int& error_code){
  int32_t *dev_a = a, *dev_b = b, *dev_c = c, *dev_bias = bias, useBias = 0;
  if(bias != NULL) useBias = 1;

  dim3 bDim(TILE_WIDTH, TILE_WIDTH, 1);
  int gh = (m + TILE_WIDTH - 1) / TILE_WIDTH;
  int gw = (n + TILE_WIDTH - 1) / TILE_WIDTH;
  dim3 gDim(gw, gh, 1);
  kernel_dense<<<gDim, bDim>>>(dev_a, dev_b, dev_c, m, k, n, dev_bias, useBias);

  cudaError_t error = cudaGetLastError();
  if(cudaSuccess != error){
    error_code = ERROR_KERNEL;
  }
  print_to_file(dev_c, m*n, "/tmp/zkh/trec/gpu/dense.txt");
  return check_cuda_error(error);
}

__global__ void kernel_relu(const int32_t *x, int32_t*y, const uint64_t n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x * blockDim.x){
    y[i] = max(x[i], 0);
  }
}
const char* cuda_relu(const int32_t *x, int32_t *y, const uint64_t n, int& error_code){
  const int32_t *dev_x = x;
  int32_t *dev_y = y;

  int threadSize = 256;
  int blockSize = getGridSize(n, threadSize);//(n + threadSize - 1) / threadSize;
  kernel_relu<<<blockSize, threadSize>>>(dev_x, dev_y, n);

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

__global__ void kernel_concatenate(const int32_t *input, const int64_t *ishape, int32_t *output, 
    int64_t* oshape, const int32_t odim, const int64_t n,  
    const int64_t preShapeSize, const int64_t curShapeSize, const int32_t axis){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(uint64_t i = tid; i < n; i += gridDim.x*blockDim.x){
    uint64_t o_i = i, in_i2 = 0, shapeSize = 0;
    bool flag = true;
    for(int j = odim-1; j >= 0; j--){
      uint64_t col = o_i % oshape[j];
      o_i /= oshape[j];
      uint64_t tmpcol = col;
      if(j == axis){
        if(col >= preShapeSize && col < curShapeSize) {
          tmpcol = col - preShapeSize;
        }else{
          flag = false;
          break;
        }
      }
      in_i2 += (j == odim-1 ? tmpcol : tmpcol * shapeSize);
      shapeSize = (j == odim-1 ? ishape[j] : shapeSize * ishape[j]);
    }
    if(flag)
    output[i] = input[in_i2];
  }
}
const char* cuda_concatenate(const int32_t *input, const int64_t *ishape, const int32_t idim, const uint64_t in, 
    int32_t *output, int64_t* oshape, const int32_t odim, const uint64_t on,  
    const int64_t preShapeSize, const int64_t curShapeSize, const int32_t axis, int& error_code){
  const int32_t *dev_input = input;
  int32_t *dev_output = output;
  int bSize = 256;
  int gSize = getGridSize(on, bSize);//(on + bSize - 1) / bSize;

  int64_t* dev_ishape = NULL, *dev_oshape = NULL;
  cudaError_t status;
  status = cudaMalloc((void**)&dev_ishape, sizeof(int64_t) * idim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_oshape, sizeof(int64_t) * odim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMemcpy(dev_ishape, ishape, sizeof(int64_t)*idim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  status = cudaMemcpy(dev_oshape, oshape, sizeof(int64_t)*odim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  kernel_concatenate<<<gSize, bSize>>>(dev_input, dev_ishape, dev_output, dev_oshape, odim, on,
      preShapeSize, curShapeSize, axis);

  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
  }
end:
  if(dev_ishape != NULL) cudaFree(dev_ishape);
  if(dev_oshape != NULL) cudaFree(dev_oshape);
  return check_cuda_error(cudaGetLastError());
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

__global__ void kernel_repeat(const int32_t *x_data, int32_t *y_data, const int64_t *xshape,
    const int64_t *yshape, const uint64_t ysize, const int32_t ndim, const int32_t axis, 
    const int32_t repeat){
  int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){
    uint64_t o_i = i, in_i = 0, shapeSize = 0;
    for(int j = ndim-1; j >= 0; j--){
      uint64_t col = o_i % yshape[j];
      o_i /= yshape[j];
      if(j == axis) col = col / repeat;
      in_i += (j == ndim-1 ? col : col * shapeSize);
      shapeSize = (j == ndim-1 ? xshape[j] : shapeSize * xshape[j]);
    }
    y_data[i] = x_data[in_i];
  }
}
const char* cuda_repeat(const int32_t *x_data, int32_t *y_data, const int64_t *xshape,
    const int64_t *yshape, const uint64_t ysize, const int32_t xndim, const int32_t yndim, 
    const int32_t axis, const int32_t repeat, int& error_code){
  int bSize = 256;
  int gSize = getGridSize(ysize, bSize);//(ysize + bSize - 1) / bSize;
  int64_t *dev_xshape = NULL, *dev_yshape = NULL;
  cudaError_t status;
  status = cudaMalloc((void**)&dev_xshape, sizeof(int64_t) * xndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_yshape, sizeof(int64_t) * yndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMemcpy(dev_xshape, xshape, sizeof(int64_t) * xndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  status = cudaMemcpy(dev_yshape, yshape, sizeof(int64_t) * yndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }

  kernel_repeat<<<gSize, bSize>>>(x_data, y_data, dev_xshape, dev_yshape, ysize, yndim, axis, repeat);

  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
  }
end:
  if(dev_xshape != NULL) cudaFree(dev_xshape);
  if(dev_yshape != NULL) cudaFree(dev_yshape);
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

  for(int i = 0; i < batch; i++){
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
    const int64_t *xshape, const int64_t *yshape){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){
    uint64_t o_i = i, in_i = 0, shapeSize = 1;
    for(int j = xndim-1; j >= 0; j--){
      int yj = j + yndim - xndim;
      int col = o_i % yshape[yj];
      o_i /= yshape[yj];
      col = col % xshape[j];
      in_i += col * shapeSize;
      shapeSize = shapeSize * xshape[j];
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

  int threadSize = 256;
  int blockSize = getGridSize(tmp_y_size, threadSize);//(tmp_y_size + threadSize - 1) / threadSize;
  uint64_t othery = 1;
  int64_t *dev_xshape = NULL, *dev_yshape = NULL;
  cudaError_t status;
  status = cudaMalloc((void**)&dev_xshape, sizeof(int64_t) * xndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_yshape, sizeof(int64_t) * yndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMemcpy(dev_xshape, xshape, sizeof(int64_t) * xndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  status = cudaMemcpy(dev_yshape, yshape, sizeof(int64_t) * yndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }

  kernel_tile<<<blockSize, threadSize>>>(x_data, y_data, tmp_y_size, yndim, xndim, dev_xshape, dev_yshape);

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
  if(dev_xshape != NULL) cudaFree(dev_xshape);
  if(dev_yshape != NULL) cudaFree(dev_yshape);
  return check_cuda_error(cudaGetLastError());
}

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

__global__ void kernel_transpose(const int32_t *x_data, const int64_t *axes_data, int32_t *y_data, 
    const int64_t *xshape, const int64_t *yshape, const int32_t ndim, const int64_t ysize, 
    const int32_t axes_ndim){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){
    uint64_t in_i = 0, o_i = i;
    for(int j = ndim-1; j >= 0; j--){
      uint64_t col = o_i % yshape[j];
      o_i /= yshape[j];
      int xj = j;
      if(axes_ndim > 0){
        xj = axes_data[j];
      }else{
        xj = ndim - 1 - j;
      }
      int xi = 1;
      for(int tx = ndim-1; tx > xj; tx--){
        xi *= xshape[tx];
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
  int64_t *dev_xshape = NULL, *dev_yshape = NULL, *dev_axes = NULL;
  cudaError_t status;
  status = cudaMalloc((void**)&dev_xshape, sizeof(int64_t) * ndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_yshape, sizeof(int64_t) * ndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMemcpy(dev_xshape, xshape, sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  status = cudaMemcpy(dev_yshape, yshape, sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  if(axes_ndim > 0){
    status = cudaMalloc((void**)&dev_axes, sizeof(int64_t) * axes_ndim);
    if(status != cudaSuccess){
      error_code = ERROR_MALLOC;
      goto end;
    }
    status = cudaMemcpy(dev_axes, axes_data, sizeof(int64_t) * axes_ndim, cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
      error_code = ERROR_MEMCPY;
      goto end;
    }
  }

  kernel_transpose<<<blockSize, threadSize>>>(x_data, dev_axes, y_data, dev_xshape, dev_yshape, ndim, ysize, axes_ndim);
  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
  }

end:
  if(dev_xshape != NULL) cudaFree(dev_xshape);
  if(dev_yshape != NULL) cudaFree(dev_yshape);
  if(axes_ndim > 0){
    if(dev_axes != NULL) cudaFree(dev_axes);
  }
  return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_stride_slice(const int32_t *x_data, int32_t *y_data, const int64_t *begin_data,
    const int32_t begin_ndim, const int64_t *step_data, const int64_t *xshape, const int64_t *yshape, 
    const int32_t step_ndim, const int32_t y_ndim, const uint64_t ysize){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(uint64_t i = tid; i < ysize; i += gridDim.x*blockDim.x){
    uint64_t o_i = i, in_i = 0, shapeSize = 0;
    for(int j = y_ndim-1; j >= 0; j--){
      uint64_t col = o_i % yshape[j];
      o_i /= yshape[j];
      int64_t begin = begin_ndim > j ? begin_data[j] : 0;
      int64_t step = step_ndim > j ? step_data[j] : 1;
      col = begin + col * step;
      in_i += (j == y_ndim-1 ? col : col * shapeSize);
      shapeSize = (j == y_ndim-1 ? xshape[j] : shapeSize * xshape[j]);
    }
    y_data[i] = x_data[in_i];
  }
}
const char* cuda_stride_slice(const int32_t *x_data, int32_t *y_data, const int64_t *begin_data,
    const int32_t begin_ndim, const int64_t *step_data, const int64_t *xshape, const int64_t *yshape, 
    const int32_t step_ndim, const int32_t y_ndim, const uint64_t ysize, const int32_t x_ndim, int& error_code){
  int threadSize = 256;
  int blockSize = getGridSize(ysize, threadSize);
  int64_t *dev_xshape = NULL, *dev_yshape = NULL, *dev_begin = NULL, *dev_step = NULL;
  cudaError_t status;
  status = cudaMalloc((void**)&dev_xshape, sizeof(int64_t) * x_ndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_yshape, sizeof(int64_t) * y_ndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_begin, sizeof(int64_t) * begin_ndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMemcpy(dev_xshape, xshape, sizeof(int64_t) * x_ndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  status = cudaMemcpy(dev_yshape, yshape, sizeof(int64_t) * y_ndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  status = cudaMemcpy(dev_begin, begin_data, sizeof(int64_t) * begin_ndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  if(step_ndim > 0){
    status = cudaMalloc((void**)&dev_step, sizeof(int64_t) * step_ndim);
    if(status != cudaSuccess){
      error_code = ERROR_MALLOC;
      goto end;
    }
    status = cudaMemcpy(dev_step, step_data, sizeof(int64_t) * step_ndim, cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
      error_code = ERROR_MEMCPY;
      goto end;
    }
  }

  kernel_stride_slice<<<blockSize, threadSize>>>(x_data,  y_data, dev_begin, begin_ndim, dev_step, 
      dev_xshape, dev_yshape, step_ndim, y_ndim, ysize);
  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
  }

end:
  if(dev_xshape != NULL) cudaFree(dev_xshape);
  if(dev_yshape != NULL) cudaFree(dev_yshape);
  if(dev_begin != NULL) cudaFree(dev_begin);
  if(step_ndim > 0){
    if(dev_step != NULL) cudaFree(dev_step);
  }
  return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_slice_like(const int32_t *x_data, int32_t *y_data, const int64_t *xshape, const int64_t *yshape,
    const uint64_t ysize, const int32_t ndim){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(uint64_t i = tid; i < ysize; i+=gridDim.x*blockDim.x){
    uint64_t o_i = i, in_i = 0, shapeSize = 1;
    for(int j = ndim-1; j >= 0; j--){
      int col = o_i % yshape[j];
      o_i /= yshape[j];
      in_i +=  col * shapeSize;
      shapeSize = shapeSize * xshape[j];
    }
    y_data[i] = x_data[in_i];
  }
}
const char* cuda_slice_like(const int32_t *x_data, int32_t *y_data, const int64_t *xshape, const int64_t *yshape,
    const uint64_t ysize, const int32_t ndim, int& error_code){
  int threadSize = 256;
  int blockSize = getGridSize(ysize, threadSize);//(ysize + threadSize - 1) / threadSize;
  int64_t *dev_xshape = NULL, *dev_yshape = NULL;
  cudaError_t status;
  status = cudaMalloc((void**)&dev_xshape, sizeof(int64_t) * ndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_yshape, sizeof(int64_t) * ndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMemcpy(dev_xshape, xshape, sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }
  status = cudaMemcpy(dev_yshape, yshape, sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }

  kernel_slice_like<<<blockSize, threadSize>>>(x_data, y_data, dev_xshape, dev_yshape, ysize, ndim);
  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
  }
end:
  if(dev_xshape != NULL) cudaFree(dev_xshape);
  if(dev_yshape != NULL) cudaFree(dev_yshape);
  return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_take(const int32_t *x_data, const int32_t *indices_data, int32_t *y_data, 
    const int64_t *xshape, const int64_t *yshape, const int64_t *indices_shape, const int32_t yndim,
    const int32_t xndim, const int32_t indices_ndim, const uint64_t ysize, const int32_t axis){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(uint64_t i = tid; i < ysize; i += gridDim.x*blockDim.x){
    uint64_t o_i = i, x_i = 0, indices_i = 0, x_shape_size = 0, indices_shape_size = 0;
    for(int32_t j = yndim - 1, k = indices_ndim-1; j>=axis; j--){
      uint64_t col = o_i % yshape[j];
      o_i /= yshape[j];
      if(j < axis + indices_ndim){
        indices_i += (indices_shape_size == 0 ? col : col * indices_shape_size);
        indices_shape_size = (indices_shape_size == 0 ? indices_shape[k]
            : indices_shape_size * indices_shape[k]);
        --k;
      }
    }

    o_i = i;
    int32_t k = xndim - 1;
    for(int32_t j = yndim - 1; j >= axis + indices_ndim; j--, k--){
      uint64_t col = o_i % yshape[j];
      o_i /= yshape[j];
      x_i += (j == yndim-1 ? col : col * x_shape_size);
      x_shape_size = (j == yndim-1 ? xshape[k] : x_shape_size * xshape[k]);
    }

    uint64_t x_indices_i = min(max(indices_data[indices_i], 0), (int32_t)xshape[k]-1);
    x_i += (x_shape_size == 0 ? x_indices_i : x_indices_i * x_shape_size);
    x_shape_size = (x_shape_size == 0 ? xshape[k] : x_shape_size * xshape[k]);
    --k;

    o_i = i;
    for(int32_t j = yndim - 1; j>=0 && k >= 0; j--){
      uint64_t col = o_i % yshape[j];
      o_i /= yshape[j];
      if(j < axis){
        x_i += x_shape_size == 0 ? col : col * x_shape_size;
        x_shape_size = x_shape_size == 0 ? xshape[k] : x_shape_size * xshape[k];
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
  int64_t *dev_xshape, *dev_yshape, *dev_indices_shape;
  cudaError_t status;
  status = cudaMalloc((void**)&dev_xshape, sizeof(int64_t) * xndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_yshape, sizeof(int64_t) * yndim);
  if(status != cudaSuccess){
    error_code = ERROR_MALLOC;
    goto end;
  }
  status = cudaMalloc((void**)&dev_indices_shape, sizeof(int64_t) * indices_ndim);
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
  status = cudaMemcpy(dev_indices_shape, indices_shape, sizeof(int64_t)*indices_ndim, cudaMemcpyHostToDevice);
  if(status != cudaSuccess){
    error_code = ERROR_MEMCPY;
    goto end;
  }

  kernel_take<<<blockSize, threadSize>>>(x_data, indices_data, y_data, dev_xshape, dev_yshape, dev_indices_shape,
      yndim, xndim, indices_ndim, ysize, axis);

  if(cudaSuccess != cudaGetLastError()){
    error_code = ERROR_KERNEL;
  }
  print_to_file(x_data, ysize, "/tmp/zkh/trec/gpu/take_x.txt");
  print_to_file(y_data, ysize, "/tmp/zkh/trec/gpu/take.txt");
end:
  if(dev_xshape != NULL) cudaFree(dev_xshape);
  if(dev_yshape != NULL) cudaFree(dev_yshape);
  if(dev_indices_shape != NULL) cudaFree(dev_indices_shape);
  return check_cuda_error(cudaGetLastError());
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
