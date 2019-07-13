#ifndef CVM_DLOPEN_H
#define CVM_DLOPEN_H

#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "cvm/c_api.h"

void* plugin_open(const char* path, char** err) {
	void* lib = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
	if (lib == NULL) {
		*err = (char*)dlerror();
	}
	return lib;
}

#define TYPE(name) name##_t
#define FUNC_BEGIN(name, params...) \
  typedef int (*TYPE(name)) (params); \
  int dl_##name(void *lib, params) { \
    void *func_ptr = dlsym(lib, #name); \
    if (func_ptr == NULL) { \
      printf("cannot find symbol from library: %s\n", #name); \
      return ERROR_RUNTIME; \
    } \
    TYPE(name) func = (TYPE(name))func_ptr; 
#define FUNC_END(params_name...) return func(params_name); }

FUNC_BEGIN(CVMAPILoadModel, const char* json, int json_strlen,
                            const char *param_bytes, int param_strlen,
                            void **net,
                            int device_type, int device_id);
FUNC_END(json, json_strlen, param_bytes, param_strlen,
         net, device_type, device_id);

FUNC_BEGIN(CVMAPIFreeModel, void *net);
FUNC_END(net);

FUNC_BEGIN(CVMAPIInference, void *net,
                          char *input_data, int input_len,
                          char *output_data);
FUNC_END(net, input_data, input_len, output_data);
 
FUNC_BEGIN(CVMAPIGetVersion, void *net, char *version);
FUNC_END(net, version);
FUNC_BEGIN(CVMAPIGetPreprocessMethod, void *net, char *method);
FUNC_END(net, method);
 
FUNC_BEGIN(CVMAPIGetInputLength, void *net, unsigned long long *size);
FUNC_END(net, size);
FUNC_BEGIN(CVMAPIGetOutputLength, void *net, unsigned long long *size);
FUNC_END(net, size);
FUNC_BEGIN(CVMAPIGetInputTypeSize, void *net, unsigned long long *size);
FUNC_END(net, size);
FUNC_BEGIN(CVMAPIGetOutputTypeSize, void *net, unsigned long long *size);
FUNC_END(net, size);

FUNC_BEGIN(CVMAPIGetStorageSize, void *net, unsigned long long *gas);
FUNC_END(net, gas);
FUNC_BEGIN(CVMAPIGetGasFromModel, void *net, unsigned long long *gas);
FUNC_END(net, gas);
FUNC_BEGIN(CVMAPIGetGasFromGraphFile, char *graph_json, unsigned long long *gas);
FUNC_END(graph_json, gas);

#endif // CVM_DLOPEN_H
