#ifndef OPENCL_OPS_H
#define OPENCL_OPS_H

#include <string>

#include <cvm/dlpack.h>
#include <cvm/runtime/device_api.h>
#include "../devapi/opencl_device_api.h"

const std::string kernel_str = R"(
  __kernel void elemwise_add(__global const int* a, __global const int* b, __global int *c, int n){
    int gid = get_global_id(0);
    if(gid < n){
      c[gid] = a[gid] + b[gid];
    }
  } 
)";

const DLContext ctx = {kDLOpenCL, 0};

cvm::runtime::OpenCLDeviceAPI *openclDeviceAPI = NULL;
void init(){
  static bool is_init = false;
  if(!is_init){
    openclDeviceAPI = (cvm::runtime::OpenCLDeviceAPI*)cvm::runtime::DeviceAPI::Get(ctx);
    openclDeviceAPI->CompileProgram(kernel_str);
  }
}

cl_kernel get_kernel(const char* kernel_name){
  cl_int ret;
  cl_kernel clkernel = clCreateKernel(openclDeviceAPI->program, kernel_name, &ret);
  OPENCL_CHECK_ERROR(ret);
  return clkernel; 
}

void exe_kernel(cl_kernel kernel, int32_t n){
  //clEnqueueTask(openclDeviceAPI->queue, kernel,0, NULL, NULL);
  size_t local_size = 256;
  size_t global_size = (n + 255) / 256 * 256;
  clEnqueueNDRangeKernel(openclDeviceAPI->queue, kernel,1, NULL, &global_size, &local_size, 0, NULL, NULL); 
}

void opencl_elemwise_add(void *a, void *b, void *c, uint64_t n){
  init();
  cl_kernel kernel = get_kernel("elemwise_add");
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c);
  clSetKernelArg(kernel, 3, sizeof(int), (void*)&n);

  exe_kernel(kernel, n);
}
#endif
