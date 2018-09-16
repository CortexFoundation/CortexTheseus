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

int main()
{
    FILE *fp = fopen("data/img.bin", "rb");
    char *img_data = (char*)calloc(28*28, sizeof(char));
    fread(img_data, sizeof(char), 28*28, fp);
    fclose(fp);

    void * model = load_model("cfg/mnist.cfg", "mnist_model.bin");

    printf("--------------------start predict--------------------\n");
    for (int i = 0;i<10;i++){
        int length = get_output_length(model);
        char *prediction = calloc(length, sizeof(char));
        predict(model, img_data, prediction);
        int_print_list(prediction, length, "int_prediction");
    }
    return 0;
}
