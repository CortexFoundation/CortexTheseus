#ifndef INT_BATCHNORM_LAYER_H
#define INT_BATCHNORM_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_int_batchnorm_layer(int batch, int w, int h, int c);

#ifdef GPU
void forward_int_batchnorm_layer_gpu(layer l, network net);
void pull_int_batchnorm_layer(layer l);
void push_int_batchnorm_layer(layer l);
#endif

#endif
