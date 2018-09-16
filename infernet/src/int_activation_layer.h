#ifndef INT_ACTIVATION_LAYER_H
#define INT_ACTIVATION_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_int_activation_layer(int batch, int inputs, ACTIVATION activation);

#ifdef GPU
void forward_int_activation_layer_gpu(layer l, network net);
#endif

#endif

