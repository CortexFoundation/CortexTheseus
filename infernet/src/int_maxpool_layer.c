#include "int_maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

int_maxpool_layer make_int_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    int_maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  (float*)calloc(output_size, sizeof(char));
    #ifdef GPU
    l.forward_gpu = int_forward_maxpool_layer_gpu;
    l.indexes_gpu = int32_cuda_make_array(NULL, output_size);
    l.output_gpu  = (float*)int_cuda_make_array((char*)l.output, output_size);
    #endif
#ifdef DEBUG 
     fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c); 
#endif
    return l;
}
