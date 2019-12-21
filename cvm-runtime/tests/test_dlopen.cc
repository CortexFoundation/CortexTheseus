#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <cstring>

static void* plugin_open(const char* path, char** err) {
	void* lib = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
	if (lib == NULL) {
		*err = (char*)dlerror();
	}
	return lib;
}

const int SUCCEED = 0;
const int ERROR_LOGIC = 1;
const int ERROR_RUNTIME = 2;

// template<typename Func_Type, typename... Args>
// int func_runner(void *lib, const char* name, Args&&... args) {
// 	void* func_ptr = dlsym(lib, name);
//   if (func_ptr == NULL) return ERROR_RUNTIME;
//   Func_Type &fn = reinterpret_cast<Func_Type>(func_ptr);
//   return fn(std::forward<Args>(args)...);
// }

// auto load_model = func_runner<API_LoadModel>;

typedef int (*Fn_LoadModel) (const char*, int, const char*, int, void**, int, int);
int API_LoadModel(void *lib, const char* json, int json_strlen,
                  const char *param, int param_strlen,
                  void **net,
                  int device_type, int device_id) {
  void *func_ptr = dlsym(lib, "CVMAPILoadModel");
  if (func_ptr == NULL) return ERROR_RUNTIME;
  Fn_LoadModel func = (Fn_LoadModel)func_ptr;
  return func(json, json_strlen, param, param_strlen, net, device_type, device_id);
}

int main() {
  const char* const libpath = "./libcvm_runtime_cpu.so";
  char *cErr;
  void *lib = plugin_open(libpath, &cErr);
  std::string symbol("{}");
  std::string params("dksjleifsldjke");
  void *net = NULL;
  int status = API_LoadModel(lib, symbol.data(), symbol.size(),
                             params.data(), params.size(),
                             &net, 0, 0);
  printf("LoadModel succeed: %d\n", status);
  return 0;
}
