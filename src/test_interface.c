#include "stdio.h"
#include "stdlib.h"
#include "interface.h"

void int_print_list(char* x, int n, char* name){
    printf("%s: ", name);
    for (int i = 0; i < n; ++i) {
        printf("%d ", x[i]);
    }
    printf("\n");
}
// #define DOG_CAT
// #define COMPRESS
// #define RES
int main()
{
#ifdef DOG_CAT
    #ifdef COMPRESS
        FILE *fp = fopen("data/dog.bin", "rb");
        char *img_data = (char*)calloc(224*224*3, sizeof(char));
        // fread(img_data, sizeof(char), 224*224*3, fp);
        fread(img_data, sizeof(char), 224*224*3, fp);
        void * model = load_model("cfg/dog_cat_compress_v1.cfg", "dog_cat_compress_v1.bin");
        if (!model) return 1;
        printf("--------------------start predict--------------------\n");
        for (int i = 0;i<500;i++){
            // fread(img_data, sizeof(char), 224*224*3, fp);
            int length = get_output_length(model);
            char *prediction = calloc(length, sizeof(char));
            predict(model, img_data, prediction);
            int_print_list(prediction, length, "int_prediction");
        }
        fclose(fp);
        return 0;
    #else
        FILE *fp = fopen("data/dog.bin", "rb");
        char *img_data = (char*)calloc(224*224*3, sizeof(char));
        // fread(img_data, sizeof(char), 224*224*3, fp);
        fread(img_data, sizeof(char), 224*224*3, fp);
        void * model = load_model("cfg/dog_cat_v1.cfg", "dog_cat_v1.bin");
        if (!model) return 1;
        printf("--------------------start predict--------------------\n");
        for (int i = 0;i<500;i++){
            // fread(img_data, sizeof(char), 224*224*3, fp);
            int length = get_output_length(model);
            char *prediction = calloc(length, sizeof(char));
            predict(model, img_data, prediction);
            int_print_list(prediction, length, "int_prediction");
        }
        fclose(fp);
        return 0;
    #endif
#else
    #ifdef RES
    FILE *fp = fopen("data/img.bin", "rb");
    char *img_data = (char*)calloc(28*28, sizeof(char));
    fread(img_data, sizeof(char), 28*28, fp);
    fclose(fp);

    void * model = load_model("cfg/mnist_res_v1.cfg", "mnist_res_v1.bin");
    if (!model) return 1;
    printf("--------------------start predict--------------------\n");
    for (int i = 0;i<2;i++){
        int length = get_output_length(model);
        char *prediction = calloc(length, sizeof(char));
        predict(model, img_data, prediction);
        int_print_list(prediction, length, "int_prediction");
    }
    return 0;
    #else
    FILE *fp = fopen("data/img.bin", "rb");
    char *img_data = (char*)calloc(28*28, sizeof(char));
    fread(img_data, sizeof(char), 28*28, fp);
    fclose(fp);

    void * model = load_model("cfg/mnist_v2.cfg", "mnist_v2.bin");
    if (!model) return 1;
    printf("--------------------start predict--------------------\n");
    for (int i = 0;i<10;i++){
        int length = get_output_length(model);
        char *prediction = calloc(length, sizeof(char));
        predict(model, img_data, prediction);
        int_print_list(prediction, length, "int_prediction");
    }
    return 0;
    #endif
#endif
}
