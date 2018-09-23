#include "int_batchnorm_layer.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>

layer make_int_batchnorm_layer(int batch, int w, int h, int c)
{
#ifdef DEBUG 
     fprintf(stderr, "Int Batch Normalization Layer: %d x %d x %d image\n", w,h,c); 
#endif
    layer l = {0};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.scales = (float*)calloc(c, sizeof(char));
    l.biases = (float*)calloc(c, sizeof(char));
    l.output = (float*)calloc(h * w * c * batch, sizeof(char));

    int i;
    for(i = 0; i < c; ++i){
        ((char*)l.scales)[i] = 1;
    }

#ifdef GPU
    l.forward_gpu = forward_int_batchnorm_layer_gpu;
    l.output_gpu =  (float*)int_cuda_make_array((char*)l.output, h * w * c * batch);
    l.biases_gpu = (float*)int_cuda_make_array((char*)l.biases, c);
    l.scales_gpu = (float*)int_cuda_make_array((char*)l.scales, c);
#endif
    return l;
}

#ifdef GPU

void forward_int_batchnorm_layer_gpu(layer l, network net)
{
    int size = l.batch * l.out_h * l.out_w * l.out_c;
    int * int32_output = int32_cuda_make_array(NULL, size);
    int_scale_add_bias_gpu(int32_output, (char*)net.input_gpu, (char*)l.scales_gpu, (char*)l.biases_gpu, l.batch, l.out_c, l.out_h*l.out_w);
    cudaScale((char*)l.output_gpu, int32_output, size, l.shift_bit);
    cuda_free((float *)int32_output);
}

void pull_int_batchnorm_layer(layer l)
{
    int_cuda_pull_array((char*)l.scales_gpu, (char*)l.scales, l.c);
    int_cuda_pull_array((char*)l.biases_gpu, (char*)l.biases, l.c);
}

void push_int_batchnorm_layer(layer l)
{
    int_cuda_push_array((char*)l.scales_gpu, (char*)l.scales, l.c);
    int_cuda_push_array((char*)l.biases_gpu, (char*)l.biases, l.c);
}

#endif
