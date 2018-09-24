#ifndef INT_MAXPOOL_LAYER_H
#define INT_MAXPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer int_maxpool_layer;

int_maxpool_layer make_int_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);

#ifdef GPU
void int_forward_maxpool_layer_gpu(int_maxpool_layer l, network net);
#endif

#endif

