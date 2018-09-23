#include "int_convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

int int_convolutional_out_height(int_convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int int_convolutional_out_width(int_convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

static size_t get_workspace_size(layer l){
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(char);
}

int_convolutional_layer make_int_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, 
                                                        ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    int_convolutional_layer l = {0};
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c/groups*n*size*size, sizeof(char));
    l.biases = calloc(n, sizeof(char));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

   
   
   
   
   
    char * weights = (char*)l.weights;
    for(i = 0; i < l.nweights; ++i) weights[i] = 127*rand_normal();
    int out_w = int_convolutional_out_width(l);
    int out_h = int_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(char));

#ifdef GPU
    l.forward_gpu = forward_int_convolutional_layer_gpu;

    if(gpu_index >= 0){
        l.weights_gpu = (float*)int_cuda_make_array((char*)l.weights, l.nweights);
        l.biases_gpu = (float*)int_cuda_make_array((char*)l.biases, n);
        l.output_gpu = (float*)int_cuda_make_array((char*)l.output, l.batch*out_h*out_w*n);
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

#ifdef DEBUG 
     fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.); 
#endif

    return l;
}

#ifdef GPU

void pull_int_convolutional_layer(layer l)
{
    int_cuda_pull_array((char*)l.weights_gpu, (char*)l.weights, l.nweights);
    int_cuda_pull_array((char*)l.biases_gpu, (char*)l.biases, l.nbiases);
}
void push_int_convolutional_layer(layer l)
{
    int_cuda_push_array((char*)l.weights_gpu, (char*)l.weights, l.nweights);
    int_cuda_push_array((char*)l.biases_gpu, (char*)l.biases, l.nbiases);
}

#endif