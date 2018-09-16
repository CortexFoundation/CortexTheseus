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
        char **pprediction;
        int length;
        predict(model, img_data,pprediction,&length);
        int_print_list(*pprediction, length, "intprediction");
    }    
    
    return 0;
}