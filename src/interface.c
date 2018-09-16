#include <assert.h>
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
    for(int i = 0; i < net->n; i++)
    {
        l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            fread((char*)l.weights, sizeof(char), l.nweights, fp);
            push_int_convolutional_layer(l);
            net->layers[i] = l;
        }
        else if(l.type == CONNECTED){
            fread((char*)l.weights, sizeof(char), l.nweights, fp);
            fread((char*)l.biases, sizeof(char), l.nbiases, fp);
            push_int_connected_layer(l);
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

char *predict(void *model, char *image_fname, char *output_data, int *length)
{
    FILE *fp = fopen(image_fname , "rb");
	assert(fp);

    char *img_data = (char*)calloc(28*28, sizeof(char));
    fread(img_data, sizeof(char), 28*28, fp);
    fclose(fp);
    network *net = (network*)model;
    output_data = int_network_predict(net, (char*)img_data);
	free(img_data);
    *length = 10;    
	return 1;
}

