#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "classifier.h"
#include "cuda.h"
#include <sys/time.h>

#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "connected_layer.h"

#include "int_convolutional_layer.h"
#include "int_batchnorm_layer.h"
#include "int_connected_layer.h"

#include "interface.h"

// void int_print_list(char* x, int n, char* name){
//     printf("%s: ", name);
// 	for (int i = 0; i < n; ++i) {
//         printf("%d ", x[i]);
// 	}
//     printf("\n");
// }

void *load_model(char *cfg_fname, char *model_bin_fname)
{
    network *net = int_parse_network_cfg(cfg_fname);
    layer l;
    FILE *fp = fopen(model_bin_fname, "rb");
    if (!fp){
        fclose(fp);
        return NULL;
    }
    for(int i = 0; i < net->n; i++)
    {
        l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            fread((char*)l.weights, sizeof(char), l.nweights, fp);
            fread((char*)l.biases, sizeof(char), l.nbiases, fp);
            push_int_convolutional_layer(l);
            net->layers[i] = l;
        }
        else if(l.type == CONNECTED){
            fread((char*)l.weights, sizeof(char), l.nweights, fp);
            fread((char*)l.biases, sizeof(char), l.nbiases, fp);
            push_int_connected_layer(l);
            net->layers[i] = l;
        }
        else if(l.type == BATCHNORM){
            fread((char*)l.scales, sizeof(char), l.c, fp);
            fread((char*)l.biases, sizeof(char), l.c, fp);
            push_int_batchnorm_layer(l);
            net->layers[i] = l;
        }
    }
    fclose(fp);
    return (void*)net;
}   

void free_model(void *model)
{
    network *net = (network*)model;
    free_network(net);
}

int get_output_length(void *model)
{
    network *net = (network*)model;
    return net->outputs;
}

int predict(void *model, char *image_data, char *output_data)
{
    network *net = (network*)model;
    char *output = int_network_predict(net, (char*)image_data);
    memcpy(output_data, output, sizeof(char)*net->outputs);
    return 0;
}

