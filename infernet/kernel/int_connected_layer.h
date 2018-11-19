#ifndef INT_CONNECTED_LAYER_H
#define INT_CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_int_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);


#ifdef GPU
void forward_int_connected_layer_gpu(layer l, network net);
void pull_int_connected_layer(layer l);
void push_int_connected_layer(layer l);
#endif

#endif

