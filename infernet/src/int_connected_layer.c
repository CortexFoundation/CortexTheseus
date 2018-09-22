#include "int_connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_int_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    layer l = {0};
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;
    l.nweights = inputs * outputs;
    l.nbiases = outputs;

    l.output = (float*)calloc(batch*outputs, sizeof(char));

    l.weights = (float*)calloc(outputs*inputs, sizeof(char));
    l.biases = (float*)calloc(outputs, sizeof(char));
    l.shift_bit = 0;
#ifdef GPU
    l.forward_gpu = forward_int_connected_layer_gpu;

    l.weights_gpu = (float*)int_cuda_make_array((char*)l.weights, outputs*inputs);
    l.biases_gpu = (float*)int_cuda_make_array((char*)l.biases, outputs);

    l.output_gpu = (float*)int_cuda_make_array((char*)l.output, outputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

#ifdef GPU

void pull_int_connected_layer(layer l)
{
    int_cuda_pull_array((char*)l.weights_gpu, (char*)l.weights, l.inputs*l.outputs);
    int_cuda_pull_array((char*)l.biases_gpu, (char*)l.biases, l.outputs);
}

void push_int_connected_layer(layer l)
{
    int_cuda_push_array((char*)l.weights_gpu, (char*)l.weights, l.inputs*l.outputs);
    int_cuda_push_array((char*)l.biases_gpu, (char*)l.biases, l.outputs);
}
void print_array1(char* a, int size){
    char* b = malloc(size*sizeof(char));
    int_cuda_pull_array(a,b,size);
    for (int i = 0;i < 10;i++){
        printf("%d,",b[i]);
    }
    printf("\n");
    free(b);   
}
void forward_int_connected_layer_gpu(layer l, network net)
{
    int_fill_gpu(l.outputs*l.batch, 0, (char*)l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    char * a = (char*)net.input_gpu;
    char * b = (char*)l.weights_gpu;
    char * c = (char*)l.output_gpu;
    char * bias = (char*)l.biases_gpu;
    int_gemm_bias_gpu(0,0,m,n,k,0,a,k,b,k,1,c,n,l.shift_bit,bias);
    int_activate_array_gpu((char*)l.output_gpu, l.outputs*l.batch, l.activation);
}

#endif
