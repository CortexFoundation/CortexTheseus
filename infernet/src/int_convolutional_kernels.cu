#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "int_convolutional_layer.h"
#include "int_activation_layer.h"

#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

void forward_int_convolutional_layer_gpu(int_convolutional_layer l, network net)
{
    int_fill_gpu(l.outputs*l.batch, 0, (char*)l.output_gpu, 1);

    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            char *a = (char*)l.weights_gpu + j*l.nweights/l.groups;
            char *b = (char*)net.workspace;
            char *c = (char*)l.output_gpu + (i*l.groups + j)*n*m;
            char *im = (char*)net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                int_im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            int_gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n, l.shift_bit);
            
        }
    }
    
    if (l.batch_normalize) {
       
       
    } else {
        int_add_bias_gpu((char*)l.output_gpu, (char*)l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    int_activate_array_gpu((char*)l.output_gpu, l.outputs*l.batch, l.activation);
   
}