/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#ifndef CVM_OPENCL_DEVICE_API_H
#define CVM_OPENCL_DEVICE_API_H

#include <cvm/runtime/device_api.h>

#include <utils/thread_local.h>
#include <cvm/runtime/registry.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

inline const char* CLGetErrorString(cl_int error) {
  switch (error) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    default: return "Unknown OpenCL error code";
  }
}
#define OPENCL_CHECK_ERROR(e)                                           \
  {                                                                     \
    CHECK(e == CL_SUCCESS)                                              \
        << "OpenCL Error, code=" << e << ": " << CLGetErrorString(e); \
  }

#define OPENCL_CALL(func)                                             \
  {                                                                   \
    cl_int e = (func);                                                \
    OPENCL_CHECK_ERROR(e);                                            \
  }
#endif


namespace cvm {
namespace runtime {

class OpenCLDeviceAPI final : public DeviceAPI {
  public:
  // type key
  std::string type_key;
  // global platform id
  cl_platform_id platform_id;
  // global platform name
  std::string platform_name;
  // global context of this process
  cl_context context{nullptr};
  // whether the workspace it initialized.
  bool initialized_{false};
  // the device type
  std::string device_type;
  // the devices
  std::vector<cl_device_id> devices;
  int device_id;
  // the queues
  std::vector<cl_command_queue> queues;
  cl_command_queue queue;
  cl_program program;
  // Number of registered kernels
  // Used to register kernel into the workspace.
  size_t num_registered_kernels{0};
  // The version counter, used
  size_t timestamp{0};
  // Ids that are freed by kernels.
  std::vector<size_t> free_kernel_ids;
  // the mutex for initialization
  std::mutex mu;

  bool IsOpenCLDevice(CVMContext ctx) {
    return ctx.device_type == kDLOpenCL;
  }

  cl_command_queue GetQueue(CVMContext ctx) {
    CHECK(IsOpenCLDevice(ctx));
    this->Init();
    CHECK(ctx.device_id >= 0  && static_cast<size_t>(ctx.device_id) < queues.size())
        << "Invalid OpenCL device_id=" << ctx.device_id;
    queue = queues[ctx.device_id];
    return queues[ctx.device_id];
  }

  void Init(const std::string& type_key, const std::string& device_type,
                           const std::string& platform_name = ""); 

  virtual void Init() {
    Init("opencl", "gpu");
  }

  void SetDevice(CVMContext ctx) final {
    //context.device_id = ctx.device_id;
    device_id = ctx.device_id;
  }
  void GetAttr(CVMContext ctx, DeviceAttrKind kind, CVMRetValue* rv) final; 
  void* AllocDataSpace(
      CVMContext ctx, size_t size, size_t alignment, CVMType type_hint); 
  void FreeDataSpace(CVMContext ctx, void* ptr); 
  void CopyDataFromTo(const void* from,
      size_t from_offset,
      void* to,
      size_t to_offset,
      size_t size,
      CVMContext ctx_from,
      CVMContext ctx_to,
      CVMType type_hint,
      CVMStreamHandle stream);
  static const std::shared_ptr<OpenCLDeviceAPI>& Global() {
    static std::shared_ptr<OpenCLDeviceAPI> inst = std::make_shared<OpenCLDeviceAPI>();
    return inst;
  }

  void CompileProgram(const std::string& source_code){
    cl_int ret;
    const char* source = source_code.c_str();
    size_t size = strlen(source);
    program = clCreateProgramWithSource(context, 1, (const char**)&source, (const size_t*)&size, &ret);         
    OPENCL_CHECK_ERROR(ret);
    OPENCL_CALL(clBuildProgram(program, 1, &devices[device_id], NULL, NULL, NULL)); 
  }
};

}
}
