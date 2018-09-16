#ifndef INTERFACE
#define INTERFACE
void *load_model(char *cfg_fname, char *model_bin_fname);
void free_model(void *net);
int get_output_length(void *net);
int predict(void *model, char *image_data, char *output_data);
#endif
