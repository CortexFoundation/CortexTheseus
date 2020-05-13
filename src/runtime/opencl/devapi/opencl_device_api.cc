#include "opencl_device_api.h"

namespace cvm {
namespace runtime {



  std::string GetPlatformInfo(
      cl_platform_id pid, cl_platform_info param_name) {
    size_t ret_size;
    OPENCL_CALL(clGetPlatformInfo(pid, param_name, 0, nullptr, &ret_size));
    std::string ret;
    ret.resize(ret_size);
    OPENCL_CALL(clGetPlatformInfo(pid, param_name, ret_size, &ret[0], nullptr));
    return ret;
  }

  bool MatchPlatformInfo(
      cl_platform_id pid,
      cl_platform_info param_name,
      std::string value) {
    if (value.length() == 0) return true;
    std::string param_value = GetPlatformInfo(pid, param_name);
    return param_value.find(value) != std::string::npos;
  }
  std::vector<cl_platform_id> GetPlatformIDs() {
    cl_uint ret_size;
    cl_int code = clGetPlatformIDs(0, nullptr, &ret_size);
    std::vector<cl_platform_id> ret;
    if (code != CL_SUCCESS) return ret;
    ret.resize(ret_size);
    OPENCL_CALL(clGetPlatformIDs(ret_size, &ret[0], nullptr));
    return ret;
  }
  std::vector<cl_device_id> GetDeviceIDs(
      cl_platform_id pid, std::string device_type) {
    cl_device_type dtype = CL_DEVICE_TYPE_ALL;
    if (device_type == "cpu") dtype = CL_DEVICE_TYPE_CPU;
    if (device_type == "gpu") dtype = CL_DEVICE_TYPE_GPU;
    if (device_type == "accelerator") dtype = CL_DEVICE_TYPE_ACCELERATOR;
    cl_uint ret_size;
    cl_int code = clGetDeviceIDs(pid, dtype, 0, nullptr, &ret_size);
    std::vector<cl_device_id> ret;
    if (code != CL_SUCCESS) return ret;
    ret.resize(ret_size);
    OPENCL_CALL(clGetDeviceIDs(pid, dtype, ret_size, &ret[0], nullptr));
    return ret;
  }

  void OpenCLDeviceAPI::Init(const std::string& type_key, const std::string& device_type,
                           const std::string& platform_name) {
    if (initialized_) return;
    std::lock_guard<std::mutex> lock(this->mu);
    if (initialized_) return;
    if (context != nullptr) return;
    this->type_key = type_key;
    // matched platforms
    std::vector<cl_platform_id> platform_ids = GetPlatformIDs();
    if (platform_ids.size() == 0) {
      LOG(WARNING) << "No OpenCL platform matched given existing options ...";
      return;
    }
    this->platform_id = nullptr;
    for (auto platform_id : platform_ids) {
      if (!MatchPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_name)) {
        continue;
      }
      std::vector<cl_device_id> devices_matched = GetDeviceIDs(platform_id, device_type);
      if ((devices_matched.size() == 0) && (device_type == "gpu")) {
        LOG(WARNING) << "Using CPU OpenCL device";
        devices_matched = GetDeviceIDs(platform_id, "cpu");
      }
      if (devices_matched.size() > 0) {
        this->platform_id = platform_id;
        this->platform_name = GetPlatformInfo(platform_id, CL_PLATFORM_NAME);
        this->device_type = device_type;
        this->devices = devices_matched;
        break;
      }
    }
    if (this->platform_id == nullptr) {
      LOG(WARNING) << "No OpenCL device";
      return;
    }
    cl_int err_code;
    this->context = clCreateContext(
        nullptr, this->devices.size(), &(this->devices[0]),
        nullptr, nullptr, &err_code);
    OPENCL_CHECK_ERROR(err_code);
    CHECK_EQ(this->queues.size(), 0U);
    for (size_t i = 0; i < this->devices.size(); ++i) {
      cl_device_id did = this->devices[i];
      this->queues.push_back(
          clCreateCommandQueue(this->context, did, 0, &err_code));
      OPENCL_CHECK_ERROR(err_code);
    }
    initialized_ = true;
  }

  void OpenCLDeviceAPI::GetAttr(CVMContext ctx, DeviceAttrKind kind, CVMRetValue* rv) {
    this->Init();
    size_t index = static_cast<size_t>(ctx.device_id);
    if (kind == kExist) {
      *rv = static_cast<int>(index< devices.size());
      return;
    }
    CHECK_LT(index, devices.size())
      << "Invalid device id " << index;
    switch (kind) {
      case kExist: break;
      case kMaxThreadsPerBlock: {
                                  size_t value;
                                  OPENCL_CALL(clGetDeviceInfo(
                                        devices[index],  CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                        sizeof(size_t), &value, nullptr));
                                  *rv = static_cast<int64_t>(value);
                                  break;
                                }
      case kWarpSize: {
                        /* TODO: the warp size of OpenCL device is not always 1
                           e.g. Intel Graphics has a sub group concept which contains 8 - 32 work items,
                           corresponding to the number of SIMD entries the heardware configures.
                           We need to figure out a way to query this information from the hardware.
                           */
                        *rv = 1;
                        break;
                      }
      case kMaxSharedMemoryPerBlock: {
                                       cl_ulong value;
                                       OPENCL_CALL(clGetDeviceInfo(
                                             devices[index], CL_DEVICE_LOCAL_MEM_SIZE,
                                             sizeof(cl_ulong), &value, nullptr));
                                       *rv = static_cast<int64_t>(value);
                                       break;
                                     }
      case kComputeVersion: return;
      case kDeviceName: {
                          char value[128] = {0};
                          OPENCL_CALL(clGetDeviceInfo(
                                devices[index], CL_DEVICE_NAME,
                                sizeof(value) - 1, value, nullptr));
                          *rv = std::string(value);
                          break;
                        }
      case kMaxClockRate: {
                            cl_uint value;
                            OPENCL_CALL(clGetDeviceInfo(
                                  devices[index], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                  sizeof(cl_uint), &value, nullptr));
                            *rv = static_cast<int32_t>(value);
                            break;
                          }
      case kMultiProcessorCount: {
                                   cl_uint value;
                                   OPENCL_CALL(clGetDeviceInfo(
                                         devices[index], CL_DEVICE_MAX_COMPUTE_UNITS,
                                         sizeof(cl_uint), &value, nullptr));
                                   *rv = static_cast<int32_t>(value);
                                   break;
                                 }
      case kMaxThreadDimensions: {
                                   size_t dims[3];
                                   OPENCL_CALL(clGetDeviceInfo(
                                         devices[index], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims, nullptr));

                                   std::stringstream ss;  // use json string to return multiple int values;
                                   ss << "[" << dims[0] <<", " << dims[1] << ", " << dims[2] << "]";
                                   *rv = ss.str();
                                   break;
                                 }
    }
  }
  void* OpenCLDeviceAPI::AllocDataSpace(
      CVMContext ctx, size_t size, size_t alignment, CVMType type_hint) {
    this->Init();
    CHECK(context != nullptr) << "No OpenCL device";
    if(size == 0) return nullptr;
    cl_int err_code;
    cl_mem mptr = clCreateBuffer(
        this->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
    OPENCL_CHECK_ERROR(err_code);
    return mptr;
  }

  void OpenCLDeviceAPI::FreeDataSpace(CVMContext ctx, void* ptr) {
    // We have to make sure that the memory object is not in the command queue
    // for some OpenCL platforms.
    OPENCL_CALL(clFinish(this->GetQueue(ctx)));

    cl_mem mptr = static_cast<cl_mem>(ptr);
    OPENCL_CALL(clReleaseMemObject(mptr));
  }

  void OpenCLDeviceAPI::CopyDataFromTo(const void* from,
      size_t from_offset,
      void* to,
      size_t to_offset,
      size_t size,
      CVMContext ctx_from,
      CVMContext ctx_to,
      CVMType type_hint,
      CVMStreamHandle stream) {
    this->Init();
    CHECK(stream == nullptr);
    if (IsOpenCLDevice(ctx_from) && IsOpenCLDevice(ctx_to)) {
      OPENCL_CALL(clEnqueueCopyBuffer(
            this->GetQueue(ctx_to),
            static_cast<cl_mem>((void*)from),  // NOLINT(*)
            static_cast<cl_mem>(to),
            from_offset, to_offset, size, 0, nullptr, nullptr));
    } else if (IsOpenCLDevice(ctx_from) && ctx_to.device_type == kDLCPU) {
      OPENCL_CALL(clEnqueueReadBuffer(
            this->GetQueue(ctx_from),
            static_cast<cl_mem>((void*)from),  // NOLINT(*)
            CL_FALSE, from_offset, size,
            static_cast<char*>(to) + to_offset,
            0, nullptr, nullptr));
      OPENCL_CALL(clFinish(this->GetQueue(ctx_from)));
    } else if (ctx_from.device_type == kDLCPU && IsOpenCLDevice(ctx_to)) {
      OPENCL_CALL(clEnqueueWriteBuffer(
            this->GetQueue(ctx_to),
            static_cast<cl_mem>(to),
            CL_FALSE, to_offset, size,
            static_cast<const char*>(from) + from_offset,
            0, nullptr, nullptr));
      OPENCL_CALL(clFinish(this->GetQueue(ctx_to)));
    } else {
      LOG(FATAL) << "Expect copy from/to OpenCL or between OpenCL";
    }
  }


CVM_REGISTER_GLOBAL("device_api.opencl")
.set_body([](CVMArgs args, CVMRetValue* rv) {
    DeviceAPI* ptr = OpenCLDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}
}
