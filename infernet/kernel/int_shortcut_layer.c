#include "int_shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>

layer make_int_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
#ifdef DEBUG 
     fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c); 
#endif
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.output = calloc(l.outputs*batch, sizeof(float));;

    #ifdef GPU
    l.forward_gpu = forward_int_shortcut_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}

#ifdef GPU
void forward_int_shortcut_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    int_shortcut_gpu(l.batch, l.w, l.h, l.c, (char*)(net.layers[l.index].output_gpu), l.out_w, l.out_h, l.out_c, (char)l.alpha, (char)l.beta, (char*)l.output_gpu);
    int_activate_array_gpu((char*)l.output_gpu, l.outputs*l.batch, l.activation);
}

#endif
