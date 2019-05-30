#include "cuda_ops.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <memory>
#include <string.h>

inline int32_t getShareMemorySize(int32_t device_id){
    static int32_t sharedMemPerBlock = 0;
    if(sharedMemPerBlock == 0){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        sharedMemPerBlock = prop.sharedMemPerBlock;
    }
    return sharedMemPerBlock;
}
const char* check_cuda_error(cudaError_t error){
    if(error == cudaSuccess) return NULL;
    else return cudaGetErrorString(error);
}

__global__ void kernel_elemwise_add(int32_t *a, int32_t *b, int32_t *c, int32_t n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n)
        c[i] = a[i] + b[i];
}

const char* cuda_elemwise_add(int32_t *a, int32_t *b, int32_t *c, int32_t n, bool debug){
    int32_t *dev_a = a, *dev_b = b, *dev_c = c;
    size_t size = sizeof(int32_t) * n;
    if(debug){
        check_cuda_error(cudaMalloc((void**)&dev_a, size));
        cudaMalloc((void**)&dev_b, size);
        cudaMalloc((void**)&dev_c, size);
        cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    }
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    kernel_elemwise_add<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, n);
//    cudaDeviceSynchronize();
    if(debug){
        cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }
    return check_cuda_error(cudaGetLastError());
}

#define BS 16
#define FS 8
//template<int F_H, int F_W, int STRIDE>
__global__ void kernel_conv2d(
        const int32_t * __restrict__ input, const int32_t i_n, const int32_t i_c, const int32_t i_h, const int32_t i_w,
        const int32_t * __restrict__ filter, const int32_t f_n, const int32_t f_c, const int32_t f_h, const int32_t f_w,
        const int32_t * __restrict__ bias,
        const int32_t padding_h, const int32_t padding_w,
        const int32_t stride_h, const int32_t stride_w,
        const int32_t dilation_h, const int32_t dilation_w,
        const int32_t groups,
        int32_t *output, const int32_t o_n, const int32_t o_c, const int32_t o_h, const int32_t o_w){
//    int g_y = blockDim.y * blockIdx.y + threadIdx.y;
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
        bool debug){
    if(i_n < 1 || i_c < 1 || i_h < 1 || i_w < 1 || f_n < 1 || f_c < 1 || f_h < 1 || f_w < 1 || 
            padding_h < 0 || padding_w < 0 || stride_h < 1 || stride_w < 1 || dilation_h < 1 || dilation_w < 1 ||
            o_n < 1 || o_c < 1 || o_h < 1 || o_w < 1){
        return "error args";
    }
    int32_t *dev_i = input, *dev_f = filter, *dev_o = output, *dev_b = bias;
    size_t s_i = i_n * i_c * i_h * i_w * sizeof(int32_t);
    size_t s_f = f_n * f_c * f_h * f_w * sizeof(int32_t);
    size_t s_b = o_c * sizeof(int32_t); 
    size_t s_o = o_n * o_c * o_h * o_w * sizeof(int32_t);
    cudaEvent_t start, stop;
    if(debug){
        cudaMalloc((void**)&dev_i, s_i);
        cudaMalloc((void**)&dev_f, s_f);
        cudaMalloc((void**)&dev_o, s_o);
        cudaMemcpy(dev_i, input, s_i, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_f, filter, s_f, cudaMemcpyHostToDevice);
        if(bias != NULL){
            cudaMalloc((void**)&dev_b, s_b);
            cudaMemcpy(dev_b, bias, s_b, cudaMemcpyHostToDevice);
        }
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    int tmp_f_h = (f_h - 1) * dilation_h + 1; // for dilation, to be optimized
    int tmp_f_w = (f_w - 1) * dilation_w + 1;
    int tmp_o_h = i_h + 2 * padding_h - tmp_f_h + 1; //for stride > 1 , TODO to be optimized
    int tmp_o_w = i_w + 2 * padding_w - tmp_f_w + 1;
    int32_t totalShareMemSize = getShareMemorySize(device_id);
    size_t share_size = ((BS + tmp_f_h - 1) * (BS + tmp_f_w - 1) + f_h * f_w * FS + FS) * sizeof(int32_t);
    if(share_size < totalShareMemSize){
        int b_h = BS;
        int b_w = BS;
        int32_t g_h = o_n * ((o_c + FS - 1) / FS) * ((tmp_o_h + b_h - 1) / b_h);
        int32_t g_w = (tmp_o_w + b_w - 1) / b_w;
        dim3 bDim(b_w, b_h, 1);
        dim3 gDim(g_w, g_h, 1);
        //TODO dilation and filter size is variable, should check shared memory size
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
//    cudaDeviceSynchronize();
    if(debug){
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        double ops = (double)((double)1.0*i_n * o_c * o_h * o_w * f_h * f_w * f_c * 3.0);
        printf("gpu cal time:%.4f, %f, %.4f\n", milliseconds, ops, ops / (milliseconds / 1000.0) / 1024.0/1024.0/1024.0);
        cudaMemcpy(output, dev_o, s_o, cudaMemcpyDeviceToHost);
        cudaFree(dev_i);
        cudaFree(dev_f);
        cudaFree(dev_o);
        if(bias != NULL)
            cudaFree(dev_b);
    }
    return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_depthwise_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t stride_w,
        int32_t dilation_h, int32_t dilation_w, // TODO dilation > 1
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
    //    int g_y = blockDim.y * blockIdx.y + threadIdx.y;
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
                shared_f[i*F_W + j] = filter[l_o_c * F_H * F_W + i * F_W + j];
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
        int32_t dilation_h, int32_t dilation_w, // TODO dilation > 1
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
                int32_t l_i_w = gx * stride_w + fx * dilation_h - padding_w;
                int32_t x;
                if(l_i_h < 0 || l_i_w < 0 || l_i_h >= i_h || l_i_w >= i_w)
                    x = 0;
                else x = input[l_o_n * i_c * i_h * i_w + l_o_c * i_h * i_w + l_i_h * i_w + l_i_w];
                sum += x * filter[l_o_n * i_c * f_h * f_w + l_o_c * f_h * f_w + fy * f_w + fx];
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
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, bool debug){
    int32_t *dev_i = input, *dev_f = filter, *dev_o = output, *dev_b = bias;
    size_t s_i = i_n * i_c * i_h * i_w * sizeof(int32_t);
    size_t s_f = f_n * f_c * f_h * f_w * sizeof(int32_t);
    size_t s_b = o_c * sizeof(int32_t); 
    size_t s_o = o_n * o_c * o_h * o_w * sizeof(int32_t);
    if(debug){
        cudaMalloc((void**)&dev_i, s_i);
        cudaMalloc((void**)&dev_f, s_f);
        cudaMalloc((void**)&dev_o, s_o);
        cudaMemcpy(dev_i, input, s_i, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_f, filter, s_f, cudaMemcpyHostToDevice);
        if(bias != NULL){
            cudaMalloc((void**)&dev_b, s_b);
            cudaMemcpy(dev_b, bias, s_b, cudaMemcpyHostToDevice);
        }
    }
//    clock_t start = clock();
    int b_h = BS;
    int b_w = BS;
    int tmp_f_h = (f_h - 1) * dilation_h + 1; // for dilation, to be optimized
    int tmp_f_w = (f_w - 1) * dilation_w + 1;
    int tmp_o_h = i_h + 2 * padding_h - tmp_f_h + 1; //for stride > 1
    int tmp_o_w = i_w + 2 * padding_w - tmp_f_w + 1;
    const int32_t totalShareMemSize = getShareMemorySize(device_id);
    size_t share_size = (BS + tmp_f_h - 1) * (BS + tmp_f_w - 1) * sizeof(int32_t) + f_h * f_w * sizeof(int32_t);
    if(share_size < totalShareMemSize){
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
        int32_t g_h = o_n * o_c * ((o_h + b_h - 1) / b_h); 
        int32_t g_w = (o_w + b_w - 1) / b_w;
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
    //cudaDeviceSynchronize();
//    clock_t end = clock();
//    printf("gpu cal time: %d\n", end-start);
    if(debug){
        cudaMemcpy(output, dev_o, s_o, cudaMemcpyDeviceToHost);
        cudaFree(dev_i);
        cudaFree(dev_f);
        cudaFree(dev_o);
        if(bias != NULL)
            cudaFree(dev_b);
    }
    return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_max_pool(
        int32_t *input, int32_t i_n/*TODO i_n > 1*/, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t f_h, int32_t f_w,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t stride_w,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w){
//    int g_y = blockDim.y * blockIdx.y + threadIdx.y;
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

    int32_t max_elem = int(1)<<31; 
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

    if(l_y < F_H-1){
        for(int i = l_y; i < F_H-1; i+=min_s_y){
            if(l_i_h+min_s_y+i-l_y < 0 || i_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x >= i_w)
                shared_i[(i+min_s_y)*siw + l_x] = 0;
            else
                shared_i[(i + min_s_y)*siw + l_x] = input[(i_y + min_s_y + i - l_y) * i_w + i_x];     
        }
    }
    if(l_x < F_W-1){
        for(int i = l_x; i < F_W-1; i+= min_s_x){
            if(l_i_h < 0 || i_x+min_s_x+i-l_x < 0 || l_i_h >= i_h || i_x+min_s_x+i-l_x >= i_w)
                shared_i[l_y * siw + i+min_s_x] = 0;
            else
                shared_i[l_y * siw + i + min_s_x] = input[i_y * i_w + i_x + min_s_x + i - l_x];
        }
    }
    if(l_y < F_H-1 && l_x < F_W-1){
        for(int i = l_y; i < F_H-1; i+=min_s_y){
            for(int j = l_x; j < F_W-1; j+=min_s_x){
                if(l_i_h+min_s_y+i-l_y < 0 || i_x+min_s_x+j-l_x < 0 || l_i_h+min_s_y+i-l_y >= i_h || i_x+min_s_x+j-l_x >= i_w)
                    shared_i[(i+min_s_y) * siw + j+min_s_x] = 0;
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
        int32_t *input, int32_t i_n/*TODO i_n > 1*/, int32_t i_c, int32_t i_h, int32_t i_w,
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
        int32_t maxV = (int32_t)1 << 31;
        for(int fy = 0; fy < f_h; ++fy){
            for(int fx = 0; fx < f_w; ++fx){
                int32_t l_i_h = l_o_h * stride_h + fy  - padding_h;
                int32_t l_i_w = gx * stride_w + fx - padding_w;
                int32_t x;
                if(l_i_h < 0 || l_i_w < 0 || l_i_h >= i_h || l_i_w >= i_w)
                    x = 0;
                else x = input[l_o_n * i_c * i_h * i_w + l_o_c * i_h * i_w + l_i_h * i_w + l_i_w];
                maxV = maxV < x ? x : maxV;
            }
        }
        output[gy * o_w + gx] = maxV;
    }
}
const char* cuda_max_pool(
        int32_t *input, int32_t i_n/*TODO i_n > 1*/, int32_t i_c, int32_t i_h, int32_t i_w,
        const int32_t f_h, const int32_t f_w,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t stride_w,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, bool debug){
    int32_t *dev_i = input, *dev_o = output;
    size_t s_i = i_n * i_c * i_h * i_w * sizeof(int32_t);
    size_t s_o = o_n * o_c * o_h * o_w * sizeof(int32_t);
    if(debug){
        cudaMalloc((void**)&dev_i, s_i);
        cudaMalloc((void**)&dev_o, s_o);
        cudaMemcpy(dev_i, input, s_i, cudaMemcpyHostToDevice);
    }
    
//    clock_t start = clock();
    const int32_t totalShareMemSize = getShareMemorySize(device_id);
    size_t share_size = (BS + f_h - 1) * (BS + f_w - 1) * sizeof(int32_t);
    int b_h = BS;
    int b_w = BS;
    int tmp_o_h = i_h + 2 * padding_h - f_h + 1; //for stride > 1
    int tmp_o_w = i_w + 2 * padding_w - f_w + 1;
    if(share_size < totalShareMemSize){
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
    //cudaDeviceSynchronize();
//    clock_t end = clock();
//    printf("gpu cal time: %ld\n", end-start);
    if(debug){
        cudaMemcpy(output, dev_o, s_o, cudaMemcpyDeviceToHost);
        cudaFree(dev_i);
        cudaFree(dev_o);
    }
    return check_cuda_error(cudaGetLastError());
}

#define TILE_WIDTH 16
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
        const int m, const int k, const int n, int32_t* bias, bool debug){
    int32_t *dev_a = a, *dev_b = b, *dev_c = c, *dev_bias = bias, useBias = 0;
    size_t s_a = sizeof(int32_t) * m * k;
    size_t s_b = sizeof(int32_t) * k * n;
    size_t s_c = sizeof(int32_t) * m * n;
    size_t s_bias = sizeof(int32_t) * n;
    if(debug){
        cudaMalloc((void**)&dev_a, s_a);
        cudaMalloc((void**)&dev_b, s_b);
        cudaMalloc((void**)&dev_c, s_c);
        if(bias != NULL){
            cudaMalloc((void**)&dev_bias, s_bias);
            cudaMemcpy(dev_bias, bias, s_bias, cudaMemcpyHostToDevice);
            useBias = 1;
        }
        cudaMemcpy(dev_a, a, s_a, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, s_b, cudaMemcpyHostToDevice);
    }
    if(bias != NULL) useBias = 1;

    dim3 bDim(TILE_WIDTH, TILE_WIDTH, 1);
    int gh = (m + TILE_WIDTH - 1) / TILE_WIDTH;
    int gw = (n + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 gDim(gw, gh, 1);
    kernel_dense<<<gDim, bDim>>>(dev_a, dev_b, dev_c, m, k, n, dev_bias, useBias);
    //cudaDeviceSynchronize();
    if(debug){
        cudaMemcpy(c, dev_c, s_c, cudaMemcpyDeviceToHost);
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaFree(dev_bias);
    }
    return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_clip(const int32_t *x, int32_t *y,
        const int32_t n, const int32_t maxV, const int32_t minV){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = max(min(x[i], maxV), minV);
    }
}
const char* cuda_clip(const int32_t *x, int32_t *y, const int32_t n, const int32_t max, const int32_t min, bool debug){
    const int32_t *dev_x = x;
    int32_t *tmp_x;
    int32_t *dev_y = y;
    if(debug) {
        cudaMalloc((void**)&tmp_x, n*sizeof(int32_t));
        dev_x = tmp_x;
        cudaMalloc((void**)&dev_y, n*sizeof(int32_t));
        cudaMemcpy(tmp_x, x, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_clip<<<blockSize, threadSize>>>(dev_x, dev_y, n, max, min);
   // cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(y, dev_y, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }
    return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_relu(const int32_t *x, int32_t*y, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = max(x[i], 0);
    }
}
const char* cuda_relu(const int32_t *x, int32_t *y, const int32_t n, bool debug){
    const int32_t *dev_x = x;
    int32_t *tmp_x;
    int32_t *dev_y = y;
    if(debug) {
        cudaMalloc((void**)&tmp_x, n*sizeof(int32_t));
        dev_x = tmp_x;
        cudaMalloc((void**)&dev_y, n*sizeof(int32_t));
        cudaMemcpy(tmp_x, x, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_relu<<<blockSize, threadSize>>>(dev_x, dev_y, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(y, dev_y, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }
    return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_flatten(const int32_t *x, int32_t*y, const int32_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        y[i] = x[i];
    }
}
const char* cuda_flatten(const int32_t *x, int32_t *y, const int32_t n, bool debug){
    const int32_t *dev_x = x;
    int32_t *tmp_x;
    int32_t *dev_y = y;
    if(debug) {
        cudaMalloc((void**)&tmp_x, n*sizeof(int32_t));
        dev_x = tmp_x;
        cudaMalloc((void**)&dev_y, n*sizeof(int32_t));
        cudaMemcpy(tmp_x, x, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
    }

    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_flatten<<<blockSize, threadSize>>>(dev_x, dev_y, n);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(y, dev_y, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }
    return check_cuda_error(cudaGetLastError());
}

inline __device__ int32_t broadcast_i_index(int64_t* oshape, int o_index, int64_t* ishape, int idim){
    int index = 0;
    int allIndex = 0;
    for(int i = 0; i < idim; i++){
        int idx = idim - 1 - i;
        int ovar = o_index % oshape[idx];
        if(ovar < ishape[idx]){
            index += i == 0 ? ovar : allIndex * ovar;
        }else if(ishape[idx] == 1){
        }else{
        }
        allIndex = (i == 0 ? ishape[idim-1] : allIndex * ishape[idx]);
        o_index /= oshape[idx];
    }
    return index;
}

__global__ void kernel_broadcast_add(const int32_t *a, const int32_t *b, int32_t*c, 
        const int32_t n,
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        int ai = broadcast_i_index(cshape, i, ashape, adim);
        int bi = broadcast_i_index(cshape, i, bshape, bdim);
        c[i] = a[ai] + b[bi];
    }
}
const char* cuda_broadcast_add(const int32_t *a, const int32_t *b, int32_t* c, 
        const int32_t n, 
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim,
        bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int64_t *dev_ashape, *dev_bshape, *dev_cshape;
    cudaMalloc((void**)&dev_ashape, sizeof(int64_t) * adim);
    cudaMalloc((void**)&dev_bshape, sizeof(int64_t) * bdim);
    cudaMalloc((void**)&dev_cshape, sizeof(int64_t) * cdim);
    cudaMemcpy(dev_ashape, ashape, sizeof(int64_t) * adim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bshape, bshape, sizeof(int64_t) * bdim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cshape, cshape, sizeof(int64_t) * cdim, cudaMemcpyHostToDevice);
    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_add<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n, dev_ashape, adim, dev_bshape, bdim, dev_cshape, cdim);
    //cudaDeviceSynchronize();

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
    cudaFree(dev_ashape);
    cudaFree(dev_bshape);
    cudaFree(dev_cshape);
    return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_broadcast_sub(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n,
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim
	){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	int32_t ai = broadcast_i_index(cshape, i, ashape, adim);
	int32_t bi = broadcast_i_index(cshape, i, bshape, bdim);
        c[i] = a[ai] - b[bi];
    }
}
const char* cuda_broadcast_sub(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, 
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim,
	bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int64_t *dev_ashape, *dev_bshape, *dev_cshape;
    cudaMalloc((void**)&dev_ashape, sizeof(int64_t) * adim);
    cudaMalloc((void**)&dev_bshape, sizeof(int64_t) * bdim);
    cudaMalloc((void**)&dev_cshape, sizeof(int64_t) * cdim);
    cudaMemcpy(dev_ashape, ashape, sizeof(int64_t) * adim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bshape, bshape, sizeof(int64_t) * bdim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cshape, cshape, sizeof(int64_t) * cdim, cudaMemcpyHostToDevice);
    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_sub<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n, dev_ashape, adim, dev_bshape, bdim, dev_cshape, cdim);
    //cudaDeviceSynchronize();

    cudaFree(dev_ashape);
    cudaFree(dev_bshape);
    cudaFree(dev_cshape);

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
    return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_broadcast_mul(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n,
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim
	){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	int32_t ai = broadcast_i_index(cshape, i, ashape, adim);
	int32_t bi = broadcast_i_index(cshape, i, bshape, bdim);
        c[i] = a[ai] * b[bi];
    }
}
const char* cuda_broadcast_mul(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, 
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim,
	bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int64_t *dev_ashape, *dev_bshape, *dev_cshape;
    cudaMalloc((void**)&dev_ashape, sizeof(int64_t) * adim);
    cudaMalloc((void**)&dev_bshape, sizeof(int64_t) * bdim);
    cudaMalloc((void**)&dev_cshape, sizeof(int64_t) * cdim);
    cudaMemcpy(dev_ashape, ashape, sizeof(int64_t) * adim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bshape, bshape, sizeof(int64_t) * bdim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cshape, cshape, sizeof(int64_t) * cdim, cudaMemcpyHostToDevice);
    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_mul<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n, dev_ashape, adim, dev_bshape, bdim, dev_cshape, cdim);
    //cudaDeviceSynchronize();

    cudaFree(dev_ashape);
    cudaFree(dev_bshape);
    cudaFree(dev_cshape);
    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
    return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_broadcast_div(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n,
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim
	){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	int32_t ai = broadcast_i_index(cshape, i, ashape, adim);
	int32_t bi = broadcast_i_index(cshape, i, bshape, bdim);
        c[i] = a[ai] / b[bi];
    }
}
const char* cuda_broadcast_div(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n,
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim,
	bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int h_b;
    cudaMemcpy(&h_b, dev_b, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if(h_b == 0)
        return "error: divide by zero";

    int64_t *dev_ashape, *dev_bshape, *dev_cshape;
    cudaMalloc((void**)&dev_ashape, sizeof(int64_t) * adim);
    cudaMalloc((void**)&dev_bshape, sizeof(int64_t) * bdim);
    cudaMalloc((void**)&dev_cshape, sizeof(int64_t) * cdim);
    cudaMemcpy(dev_ashape, ashape, sizeof(int64_t) * adim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bshape, bshape, sizeof(int64_t) * bdim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cshape, cshape, sizeof(int64_t) * cdim, cudaMemcpyHostToDevice);
    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_div<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n, dev_ashape, adim, dev_bshape, bdim, dev_cshape, cdim);
    //cudaDeviceSynchronize();
    cudaFree(dev_ashape);
    cudaFree(dev_bshape);
    cudaFree(dev_cshape);

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
    return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_broadcast_right_shift(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n,
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	int32_t ai = broadcast_i_index(cshape, i, ashape, adim);
	int32_t bi = broadcast_i_index(cshape, i, bshape, bdim);
        c[i] = a[ai] >> b[bi];
    }
}
const char* cuda_broadcast_right_shift(const int32_t *a, const int32_t* b, int32_t* c, const int32_t n,
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim,
	bool debug){
    const int32_t *dev_a = a;
    const int32_t *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int64_t *dev_ashape, *dev_bshape, *dev_cshape;
    cudaMalloc((void**)&dev_ashape, sizeof(int64_t) * adim);
    cudaMalloc((void**)&dev_bshape, sizeof(int64_t) * bdim);
    cudaMalloc((void**)&dev_cshape, sizeof(int64_t) * cdim);
    cudaMemcpy(dev_ashape, ashape, sizeof(int64_t) * adim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bshape, bshape, sizeof(int64_t) * bdim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cshape, cshape, sizeof(int64_t) * cdim, cudaMemcpyHostToDevice);
    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_right_shift<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n, dev_ashape, adim, dev_bshape, bdim, dev_cshape, cdim);
    //cudaDeviceSynchronize();
    cudaFree(dev_ashape);
    cudaFree(dev_bshape);
    cudaFree(dev_cshape);

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
    return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_broadcast_left_shift(const int32_t *a, const int32_t *b, int32_t*c, const int32_t n,
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim
	){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	int32_t ai = broadcast_i_index(cshape, i, ashape, adim);
	int32_t bi = broadcast_i_index(cshape, i, bshape, bdim);
        c[i] = a[ai] << b[bi];
    }
}
const char* cuda_broadcast_left_shift(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, 
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim,
	bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int64_t *dev_ashape, *dev_bshape, *dev_cshape;
    cudaMalloc((void**)&dev_ashape, sizeof(int64_t) * adim);
    cudaMalloc((void**)&dev_bshape, sizeof(int64_t) * bdim);
    cudaMalloc((void**)&dev_cshape, sizeof(int64_t) * cdim);
    cudaMemcpy(dev_ashape, ashape, sizeof(int64_t) * adim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bshape, bshape, sizeof(int64_t) * bdim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cshape, cshape, sizeof(int64_t) * cdim, cudaMemcpyHostToDevice);
    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_left_shift<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n, dev_ashape, adim, dev_bshape, bdim, dev_cshape, cdim);
    //cudaDeviceSynchronize();
    cudaFree(dev_ashape);
    cudaFree(dev_bshape);
    cudaFree(dev_cshape);

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
    return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_broadcast_max(const int32_t *a, const int32_t *b, int32_t *c, const int32_t n,
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim
	){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n){
	int32_t ai = broadcast_i_index(cshape, i, ashape, adim);
	int32_t bi = broadcast_i_index(cshape, i, bshape, bdim);
        c[i] = a[ai] > b[bi] ? a[ai] : b[bi];
    }
}
const char* cuda_broadcast_max(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n, 
        int64_t *ashape, int32_t adim,
        int64_t *bshape, int32_t bdim,
        int64_t *cshape, int32_t cdim,
	bool debug){
    const int32_t *dev_a = a, *dev_b = b;
    int32_t *tmp_a, *tmp_b;
    int32_t *dev_c = c;
    if(debug) {
        cudaMalloc((void**)&tmp_a, n*sizeof(int32_t));
        dev_a = tmp_a;
        cudaMalloc((void**)&tmp_b, sizeof(int32_t));
        dev_b = tmp_b;
        cudaMalloc((void**)&dev_c, n*sizeof(int32_t));
        cudaMemcpy(tmp_a, a, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_b, b, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int64_t *dev_ashape, *dev_bshape, *dev_cshape;
    cudaMalloc((void**)&dev_ashape, sizeof(int64_t) * adim);
    cudaMalloc((void**)&dev_bshape, sizeof(int64_t) * bdim);
    cudaMalloc((void**)&dev_cshape, sizeof(int64_t) * cdim);
    cudaMemcpy(dev_ashape, ashape, sizeof(int64_t) * adim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bshape, bshape, sizeof(int64_t) * bdim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cshape, cshape, sizeof(int64_t) * cdim, cudaMemcpyHostToDevice);
    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_broadcast_max<<<blockSize, threadSize>>>(dev_a, dev_b, dev_c, n, dev_ashape, adim, dev_bshape, bdim, dev_cshape, cdim);
    //cudaDeviceSynchronize();
    cudaFree(dev_ashape);
    cudaFree(dev_bshape);
    cudaFree(dev_cshape);

    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t)*n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_a);
        cudaFree(dev_c);
        cudaFree(tmp_b);
    }
    return check_cuda_error(cudaGetLastError());
}

//TODO use reduce
__global__ void kernel_sum(const int32_t *x,
        const int n_batch,
        const int channels,
        const int h,
        const int w,
        int32_t *y){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int n = n_batch * channels;
    if(tid < n){
        int32_t sum = 0;
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                sum += x[tid * h * w + i * w + j];
            }
        }
        y[tid] = sum;
    }
    
}
const char* cuda_sum(
        const int32_t *x,
        const int32_t n_batch, const int32_t channels, const int32_t h, const int32_t w,
        int32_t *y,
        bool debug){
    const int32_t *dev_x = x;
    int32_t *dev_y = y;
    int32_t *tmp_x;
    if(debug){
        size_t size = n_batch * channels * h * w * sizeof(int32_t);
        cudaMalloc((void**)&tmp_x, size);
        dev_x = tmp_x;
        cudaMalloc((void**)&dev_y, n_batch * channels * sizeof(int32_t));
        cudaMemcpy(tmp_x, x, size, cudaMemcpyHostToDevice);
    }

    int n = n_batch * channels;
    int threadSize = 256;
    int blockSize = (n + threadSize - 1) / threadSize;
    kernel_sum<<<blockSize, threadSize>>>(dev_x, n_batch, channels, h, w, dev_y);
    //cudaDeviceSynchronize();
    if(debug){
        cudaMemcpy(y, dev_y, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }
    return check_cuda_error(cudaGetLastError());
}

const char* cuda_reshape(const int32_t *x, int32_t *y, int32_t n, bool debug){
    if(x == y) return NULL;
    if(debug)
		 memcpy(y, x, n * sizeof(int32_t));
    else
        cudaMemcpy(y, x, n*sizeof(int32_t), cudaMemcpyDeviceToDevice);
    return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_log(const int32_t *x, int32_t *y){
    for(int i = 0; i < 64; i++){
        int64_t tmp = (int64_t)1 << i;
        if(x[0] <= tmp){
            y[0] = i;
            return;
        }
    }
    y[0] = 64;
}
const char* cuda_log(const int32_t *x, int32_t *y, const bool debug){
    const int32_t *dev_x = x;
    int32_t *tmp_x, *dev_y = y;
    if(debug){
        cudaMalloc((void**)&tmp_x, sizeof(int32_t));
        dev_x = tmp_x;
        cudaMemcpy(tmp_x, x, sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int h_x;
    cudaMemcpy(&h_x, dev_x, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if(h_x == 0) return "error: log2(0)";

    kernel_log<<<1,1>>>(dev_x, dev_y);

    if(debug){
        cudaMemcpy(y, dev_y, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }

    return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_abs(const int32_t *x, int32_t *y, const int32_t n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n){
        y[i] = abs(x[i]);
    }
}
const char* cuda_abs(const int32_t *x, int32_t *y, const int32_t n, bool debug){
    const int32_t *dev_x = x;
    int32_t *tmp_x, *dev_y = y;
    if(debug){
        cudaMalloc((void**)&tmp_x, sizeof(int32_t) * n);
        dev_x = tmp_x;
        cudaMalloc((void**)&dev_y, sizeof(int32_t) * n);
        cudaMemcpy(tmp_x, x, sizeof(int32_t) * n, cudaMemcpyHostToDevice);
    }
    int bSize = 256;
    int gSize = (n + bSize - 1) / bSize;
    kernel_abs<<<gSize, bSize>>>(dev_x, dev_y, n);
    if(debug){
        cudaMemcpy(y, dev_y, sizeof(int32_t) * n, cudaMemcpyDeviceToHost);
        cudaFree(tmp_x);
        cudaFree(dev_y);
    }
    return check_cuda_error(cudaGetLastError());
}
__global__ void kernel_max(const int32_t *x, int32_t *y, int32_t n){
   __shared__ int32_t buf[256];
   int32_t tid = threadIdx.x;
   int32_t maxValue = (int32_t)1 << 31;
   for (int i = tid; i < n; i += blockDim.x){
       int32_t tmp = x[i];
       if(maxValue < tmp) maxValue = tmp;
   }

   buf[tid] = maxValue;
   __syncthreads();
   for(int s = 1; s < blockDim.x; s*=2){
       if((tid % (2*s)) == 0){
           int a = buf[tid];
           int b = buf[tid+s];
           buf[tid] = a > b ? a : b;
       }
       __syncthreads();
   }

   if(tid == 0) y[0] = buf[0];
}
const char* cuda_max(const int32_t *x, int32_t *y, const int32_t n, bool debug){
   const int32_t *dev_x = x;
   int32_t *tmp_x, *dev_y = y;
   if(debug){
       cudaMalloc((void**)&tmp_x, sizeof(int32_t) * n);
       dev_x = tmp_x;
       cudaMalloc((void**)&dev_y, sizeof(int32_t));
       cudaMemcpy(tmp_x, x, sizeof(int32_t)*n, cudaMemcpyHostToDevice);
   }

   kernel_max<<<1, 256>>>(dev_x, dev_y, n);

   if(debug){
       cudaMemcpy(y, dev_y, sizeof(int32_t), cudaMemcpyDeviceToHost);
       cudaFree(tmp_x);
       cudaFree(dev_y);
   }
   return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_cvm_clip(const int32_t *x, const int32_t precision, int32_t *y, const int32_t n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int minV = -((1 << (precision - 1)) - 1);
    int maxV = -minV;
    if(tid < n){
        y[tid] = max(min(x[tid], maxV), minV);
    }
}
const char* cuda_cvm_clip(const int32_t* x, const int32_t precision, int32_t *y, const int32_t n, bool debug){
    const int32_t *dev_x = x;
    int32_t *tmp_x, *dev_y = y;
    if(debug){
        cudaMalloc((void**)&tmp_x, sizeof(int32_t) * n);
        cudaMalloc((void**)&dev_y, sizeof(int32_t) * n);
        cudaMemcpy(tmp_x, x, sizeof(int32_t) * n, cudaMemcpyHostToDevice);
        dev_x = tmp_x;
    }
    int bSize = 256;
    int gSize = (n + bSize - 1) / bSize;
    kernel_cvm_clip<<<gSize, bSize>>>(dev_x, precision, dev_y, n);
    if(debug){
        cudaMemcpy(y, dev_y, sizeof(int32_t) * n, cudaMemcpyDeviceToHost);
        cudaFree(dev_y);
        cudaFree(tmp_x);
    }
    
    return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_cvm_right_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const int32_t n){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int minV = -((1 << (precision - 1)) - 1);
    int maxV = -minV;
    if(tid < n){
        int shift_a = a[tid];
        if(b == 0) c[tid] = shift_a;
        else {
            shift_a = ((shift_a >> (b - 1)) + 1 ) >> 1;
            c[tid] = max(min(shift_a, maxV), minV);
        } 
    }
}
const char* cuda_cvm_right_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const int32_t n, bool debug){
    const int32_t *dev_a = a;
    int32_t *tmp_a, *dev_c = c;
    if(debug){
        cudaMalloc((void**)&tmp_a, sizeof(int32_t) * n);
        cudaMalloc((void**)&dev_c, sizeof(int32_t) * n);
        cudaMemcpy(tmp_a, a, sizeof(int32_t) * n, cudaMemcpyHostToDevice);
        dev_a = tmp_a;
    }

    int bSize = 256;
    int gSize = (n + bSize - 1) / bSize;
    kernel_cvm_right_shift<<<gSize, bSize>>>(dev_a, b, precision, dev_c, n);
    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t) * n, cudaMemcpyDeviceToHost);
        cudaFree(dev_c);
        cudaFree(tmp_a);
    }
    return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_cvm_left_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const int32_t n){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int minV = -((1 << (precision - 1)) - 1);
    int maxV = -minV;
    if(tid < n){
        int shift_a = a[tid];
        if(b == 0) c[tid] = shift_a;
        else {
            shift_a = shift_a << b;
            c[tid] = max(min(shift_a, maxV), minV);
        } 
    }
}
const char* cuda_cvm_left_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const int32_t n, bool debug){
    const int32_t *dev_a = a;
    int32_t *tmp_a, *dev_c = c;
    if(debug){
        cudaMalloc((void**)&tmp_a, sizeof(int32_t) * n);
        cudaMalloc((void**)&dev_c, sizeof(int32_t) * n);
        cudaMemcpy(tmp_a, a, sizeof(int32_t) * n, cudaMemcpyHostToDevice);
        dev_a = tmp_a;
    }

    int bSize = 256;
    int gSize = (n + bSize - 1) / bSize;
    kernel_cvm_left_shift<<<gSize, bSize>>>(dev_a, b, precision, dev_c, n);
    if(debug){
        cudaMemcpy(c, dev_c, sizeof(int32_t) * n, cudaMemcpyDeviceToHost);
        cudaFree(dev_c);
        cudaFree(tmp_a);
    }
    return check_cuda_error(cudaGetLastError());
}

__global__ void kernel_concatenate(const int32_t *input, const int64_t *ishape, int32_t *output, 
        int64_t* oshape, const int32_t odim, const int32_t n,  
        const int64_t preShapeSize, const int64_t curShapeSize, const int32_t axis){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < n){
        int32_t o_i = i, in_i2 = 0, shapeSize = 0;
        for(int j = odim-1; j >= 0; j--){
            int64_t col = o_i % oshape[j];
            o_i /= oshape[j];
            int64_t tmpcol = col;
            if(j == axis){
                if(col >= preShapeSize && col < curShapeSize) {
                    tmpcol = col - preShapeSize;
                }else{
                    return;
                }
            }
            in_i2 += (j == odim-1 ? tmpcol : tmpcol * shapeSize);
            shapeSize = (j == odim-1 ? ishape[j] : shapeSize * ishape[j]);
        }
        output[i] = input[in_i2];
	}
}
const char* cuda_concatenate(const int32_t *input, const int64_t *ishape, const int32_t idim, const int32_t in, 
        int32_t *output, int64_t* oshape, const int32_t odim, const int32_t on,  
        const int64_t preShapeSize, const int64_t curShapeSize, const int32_t axis, bool debug){
    const int32_t *dev_input = input;
    int32_t *tmp_input, *dev_output = output;
    if(debug){
        cudaMalloc((void**)&tmp_input, sizeof(int32_t) * in);
        cudaMalloc((void**)&dev_output, sizeof(int32_t) * on);
        cudaMemcpy(tmp_input, input, sizeof(int32_t) * in, cudaMemcpyHostToDevice);
        dev_input = tmp_input;
    }

    int64_t* dev_ishape, *dev_oshape;
    cudaMalloc((void**)&dev_ishape, sizeof(int64_t) * idim);
    cudaMalloc((void**)&dev_oshape, sizeof(int64_t) * odim);
    cudaMemcpy(dev_ishape, ishape, sizeof(int64_t)*idim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_oshape, oshape, sizeof(int64_t)*odim, cudaMemcpyHostToDevice);
    int bSize = 256;
    int gSize = (on + bSize - 1) / bSize;
    kernel_concatenate<<<gSize, bSize>>>(dev_input, dev_ishape, dev_output, dev_oshape, odim, on,
            preShapeSize, curShapeSize, axis);

    cudaFree(dev_ishape);
    cudaFree(dev_oshape);

    if(debug){
        cudaMemcpy(output, dev_output, sizeof(int32_t) * on, cudaMemcpyDeviceToHost);
        cudaFree(tmp_input);
        cudaFree(dev_output);
    }
    return check_cuda_error(cudaGetLastError());
}
