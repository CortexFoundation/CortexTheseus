#ifndef INTERFACE
#define INTERFACE
void *load_model(char *cfg_fname, char *model_bin_fname);
void free_model(void *net);
char *predict(void * net, char * image_data);
#endif
