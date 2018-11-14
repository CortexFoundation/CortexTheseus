#include "int_activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_int_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;

    l.output = (float*)calloc(batch*inputs, sizeof(char));

#ifdef GPU
    l.forward_gpu = forward_int_activation_layer_gpu;
    l.output_gpu = (float*)int_cuda_make_array((char*)l.output, inputs*batch);
#endif
    l.activation = activation;
#ifdef DEBUG 
     fprintf(stderr, "Activation Layer: %d inputs\n", inputs); 
#endif
    return l;
}

#ifdef GPU

void forward_int_activation_layer_gpu(layer l, network net)
{
    int_copy_gpu(l.outputs*l.batch, (char*)net.input_gpu, 1, (char*)l.output_gpu, 1);
    int_activate_array_gpu((char*)l.output_gpu, l.outputs*l.batch, l.activation);
}

#endif
