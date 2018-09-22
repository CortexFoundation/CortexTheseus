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

void print_array(float* x, int row, int col, char* name){
    printf("--------------------");
    printf("%s", name);
    printf("--------------------\n");
    
	for (int i = 0; i < row*col; ++i) {
        if(x[i] >= 0)printf(" ");
        printf("%.2f ", x[i]);
		if ((i + 1) % col == 0) printf("\n");
	}
}

void int_print_array(char* x, int row, int col, char* name){
    printf("--------------------");
    printf("%s", name);
    printf("--------------------\n");
    
	for (int i = 0; i < row*col; ++i) {
        if(x[i] >= 0)printf(" ");
        printf("%d ", x[i]);
		if ((i + 1) % col == 0) printf("\n");
	}
}

void print_list(float* x, int n, char* name){
    printf("%s: ", name);
	for (int i = 0; i < n; ++i) {
        printf("%.2f ", x[i]);
	}
    printf("\n");
}

void int_print_list(char* x, int n, char* name){
    printf("%s: ", name);
	for (int i = 0; i < n; ++i) {
        printf("%d ", x[i]);
	}
    printf("\n");
}

void set_value(float *x, float value, int n){
    for(int i = 0; i < n; i++) x[i] = value;
}

void int_set_value(char *x, char value, int n){
    for(int i = 0; i < n; i++) x[i] = value;
}

int main()
{
    srand(2222222);
    network *net = parse_network_cfg("cfg/test_fc.cfg");

    image im;
    im.w = 1; im.h = 1; im.c = 3;
    im.data = (float*)malloc(im.w*im.h*im.c*sizeof(float));

    int_image im2;
    im2.w = 1; im2.h = 1; im2.c = 3;
    im2.data = (char*)malloc(im.w*im.h*im.c*sizeof(char));

    for (int i = 0; i < im.w*im.h*im.c; i++) 
    {
        im.data[i] = i
        im2.data[i] = i;
    }

    layer l = net->layers[net->n-1];
    layer fc_backup;
    if(l.type == CONNECTED){
        printf("--------------------fc_weights--------------------\n");
        for(int i = 0; i < l.inputs * l.outputs; i++) l.weights[i] = (int)rand_uniform(-2, 2);
        l.biases[0] = 0;
        l.biases[1] = 0;
        push_connected_layer(l);
        
        net->layers[net->n-1] = l;
        print_list(l.weights, l.outputs*l.inputs, "weights");
        print_list(l.biases, l.outputs, "biases");
        fc_backup = l;
    }
    
    print_array(im.data, im.w, im.h, "img");

    printf("--------------------start predict--------------------\n");
    float *predictions = network_predict(net, im.data);
    print_list(predictions, l.outputs, "prediction");

    printf("--------------------------------------------------load_int_network-----------------------------------------------\n");
    
    net = int_parse_network_cfg("cfg/test.cfg");

    l = net->layers[net->n-1];
    if(l.type == CONNECTED){
        printf("--------------------int_fc_weights--------------------\n");
        weights2 = (char*)l.weights;
        for(int i = 0; i < l.inputs * l.outputs; i++) weights2[i] = fc_backup.weights[i];
        l.biases[0] = 0;
        l.biases[1] = 0;
        push_int_connected_layer(l);
        
        net->layers[net->n-1] = l;
        int_print_list((char*)l.weights, l.outputs*l.inputs, "weights");
        int_print_list((char*)l.biases, l.outputs, "biases");
        fc_backup = l;
    }

    int_print_array(im2.data, im2.w, im2.h, "int_img");
    
    printf("--------------------start predict--------------------\n");
    char *int_predictions = int_network_predict(net, im2.data);
    int_print_list(int_predictions, l.outputs, "int_prediction");

    
    return 0;
}