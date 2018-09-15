#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "int_maxpool_layer.h"
#include "cuda.h"
}

__global__ void int_forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, 
    char *input, char *output, int *indexes)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int out_index = j + w*(i + h*(k + c*b));
    char max = -128;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            char val = (valid != 0) ? input[index] : -128;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}

extern "C" void int_forward_maxpool_layer_gpu(int_maxpool_layer layer, network net)
{
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;

    size_t n = h*w*c*layer.batch;

    int_forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stride, layer.size, 
        layer.pad, (char*)net.input_gpu, (char*)layer.output_gpu, layer.indexes_gpu);
    check_error(cudaPeekAtLastError());
}
