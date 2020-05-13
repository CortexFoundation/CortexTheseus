/*!
 *  Copyright (c) 2017 by Contributors
 * \file module.cc
 * \brief CVM module system
 */
#include <cvm/runtime/module.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/packed_func.h>
#include <unordered_set>
#include <cstring>

namespace cvm {
namespace runtime {

bool RuntimeEnabled(const std::string& target) {
  std::string f_name;
  if (target == "cpu") {
    return true;
  } else if (target == "cuda" || target == "gpu") {
    f_name = "device_api.gpu";
  } else if (target == "cl" || target == "opencl" || target == "sdaccel") {
    f_name = "device_api.opencl";
  } else if (target == "gl" || target == "opengl") {
    f_name = "device_api.opengl";
  } else if (target == "mtl" || target == "metal") {
    f_name = "device_api.metal";
  } else if (target == "vulkan") {
    f_name = "device_api.vulkan";
  } else if (target == "stackvm") {
    f_name = "codegen.build_stackvm";
  } else if (target == "rpc") {
    f_name = "device_api.rpc";
  } else if (target == "vpi" || target == "verilog") {
    f_name = "device_api.vpi";
  } else if (target.length() >= 5 && target.substr(0, 5) == "nvptx") {
    f_name = "device_api.gpu";
  } else if (target.length() >= 4 && target.substr(0, 4) == "rocm") {
    f_name = "device_api.rocm";
  } else if (target.length() >= 4 && target.substr(0, 4) == "llvm") {
    const PackedFunc* pf = runtime::Registry::Get("codegen.llvm_target_enabled");
    if (pf == nullptr) return false;
    return (*pf)(target);
  } else {
    LOG(FATAL) << "Unknown optional runtime " << target;
  }
  return runtime::Registry::Get(f_name) != nullptr;
}

CVM_REGISTER_GLOBAL("module._Enabled")
.set_body([](CVMArgs args, CVMRetValue *ret) {
    *ret = RuntimeEnabled(args[0]);
    });

CVM_REGISTER_GLOBAL("module._GetTypeKey")
.set_body([](CVMArgs args, CVMRetValue *ret) {
    *ret = std::string(args[0].operator Module()->type_key());
    });

}  // namespace runtime
}  // namespace cvm
