#ifndef int_convolutional_layer_H
#define int_convolutional_layer_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer int_convolutional_layer;

int_convolutional_layer make_int_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, 
                                                        ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
#ifdef GPU
void forward_int_convolutional_layer_gpu(int_convolutional_layer layer, network net);
void pull_int_convolutional_layer(layer l);
void push_int_convolutional_layer(layer l);
#endif

#endif

