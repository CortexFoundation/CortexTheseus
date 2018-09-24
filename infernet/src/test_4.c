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

// void int_print_list(char* x, int n, char* name){
//     printf("%s: ", name);
// 	for (int i = 0; i < n; ++i) {
//         printf("%d ", x[i]);
// 	}
//     printf("\n");
// }


network *net;
void load_model(char *cfg_fname, char *model_bin_fname)
{
    net = int_parse_network_cfg(cfg_fname);
    layer l;
    FILE *fp = fopen(model_bin_fname, "rb");
    for(int i = 0; i < net->n; i++)
    {
        l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            size_t ret = fread((char*)l.weights, sizeof(char), l.nweights, fp);
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
}

char *predict(char * image_data)
{
    return int_network_predict(net, (char*)image_data);
}

int main()
{
    FILE *fp = fopen("data/img.bin", "rb");
    char *img_data = calloc(28*28, sizeof(char));
    fread(img_data, sizeof(char), 28*28, fp);
    fclose(fp);
   

    load_model("cfg/mnist.cfg", "mnist_model.bin");

   
    for (int i = 0;i<100;i++){
       
        char * prediction = predict(img_data);
       
       

       
    }    
    
    return 0;
}