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

#define ARGS_COUNT(args...) ARGS_COUNT_(0, ##args, \
    16, 15, 14, 13, 12, 11, 10, \
    9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define ARGS_COUNT_( \
    _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, \
    _10, _11, _12, _13, _14, _15, _16, N, ...) N

#define ARG_0(f)
#define ARG_2(f, t, v) f(t, v)
#define ARG_4(f, t, v, args...) f(t, v), ARG_2(f, args)
#define ARG_6(f, t, v, args...) f(t, v), ARG_4(f, args)
#define ARG_8(f, t, v, args...) f(t, v), ARG_6(f, args)
#define ARG_10(f, t, v, args...) f(t, v), ARG_8(f, args)
#define ARG_12(f, t, v, args...) f(t, v), ARG_10(f, args)
#define ARG_14(f, t, v, args...) f(t, v), ARG_12(f, args)

#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)

#define T(t, v) t
#define V(t, v) v
#define TV(t, v) t v
#define ARG_T(args...) CONCAT(ARG_, ARGS_COUNT(args))(T, ##args)
#define ARG_V(args...) CONCAT(ARG_, ARGS_COUNT(args))(V, ##args)
#define ARG_TV(args...) CONCAT(ARG_, ARGS_COUNT(args))(TV, ##args)

#define MAKE_FUNC(fname, params...) \
  typedef int (*TYPE(fname)) (ARG_T(params)); \
  int dl_##fname(void *lib, ARG_TV(params)) { \
    void *func_ptr = dlsym(lib, #fname); \
    if (func_ptr == NULL) { \
      printf("cannot find symbol from library: %s\n", #fname); \
      return ERROR_RUNTIME; \
    } \
    TYPE(fname) func = (TYPE(fname))func_ptr; \
    return func(ARG_V(params)); \
  }

MAKE_FUNC(CVMAPILoadModel, const char*, json, int, json_strlen,
                           const char*, param_bytes, int, param_strlen,
                           void**, net,
                           int, device_type, int, device_id);
MAKE_FUNC(CVMAPIFreeModel, void*, net);
MAKE_FUNC(CVMAPIInference, void*, net,
                           char*, input_data, int, input_len,
                           char*, output_data);

MAKE_FUNC(CVMAPIGetVersion, void*, net, char*, version);
MAKE_FUNC(CVMAPIGetPreprocessMethod, void*, net, char*, method);

MAKE_FUNC(CVMAPIGetInputLength, void*, net, unsigned long long*, size);
MAKE_FUNC(CVMAPIGetOutputLength, void*, net, unsigned long long*, size);
MAKE_FUNC(CVMAPIGetInputTypeSize, void*, net, unsigned long long*, size);
MAKE_FUNC(CVMAPIGetOutputTypeSize, void*, net, unsigned long long*, size);

MAKE_FUNC(CVMAPIGetStorageSize, void*, net, unsigned long long*, gas);
MAKE_FUNC(CVMAPIGetGasFromModel, void*, net, unsigned long long*, gas);
MAKE_FUNC(CVMAPIGetGasFromGraphFile, const char*, graph_json, unsigned long long*, gas);

#endif // CVM_DLOPEN_H
