#ifndef INT_SHORTCUT_LAYER_H
#define INT_SHORTCUT_LAYER_H

#include "layer.h"
#include "network.h"

layer make_int_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
#ifdef GPU
void forward_int_shortcut_layer_gpu(const layer l, network net);
#endif

#endif
