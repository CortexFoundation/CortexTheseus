/*!
 *  Copyright (c) 2017 by Contributors
 * \file cpu_dsl_api.cc
 * \brief DSL API dispatcher
 */
#include <cvm/runtime/registry.h>
#include <cvm/c_dsl_api.h>
#include "dsl_api.h"
#include "runtime_base.h"

namespace cvm {
namespace runtime {

DSLAPI* FindDSLAPI() {
  auto* f = Registry::Get("dsl_api.singleton");
  if (f == nullptr) {
    throw utils::Error("CVM runtime only environment,"\
                      " DSL API is not available");
  }
  void* ptr = (*f)();
  return static_cast<DSLAPI*>(ptr);
}

static DSLAPI* GetDSLAPI() {
  static DSLAPI* inst = FindDSLAPI();
  return inst;
}
}  // namespace runtime
}  // namespace cvm

using namespace cvm::runtime;

int CVMNodeFree(NodeHandle handle) {
  API_BEGIN();
  GetDSLAPI()->NodeFree(handle);
  API_END();
}

int CVMNodeTypeKey2Index(const char* type_key,
                         int* out_index) {
  API_BEGIN();
  GetDSLAPI()->NodeTypeKey2Index(type_key, out_index);
  API_END();
}


int CVMNodeGetTypeIndex(NodeHandle handle,
                        int* out_index) {
  API_BEGIN();
  GetDSLAPI()->NodeGetTypeIndex(handle, out_index);
  API_END();
}

int CVMNodeGetAttr(NodeHandle handle,
                   const char* key,
                   CVMValue* out_value,
                   int* out_type_code,
                   int* out_success) {
  API_BEGIN();
  GetDSLAPI()->NodeGetAttr(
      handle, key, out_value, out_type_code, out_success);
  API_END();
}

int CVMNodeListAttrNames(NodeHandle handle,
                         int *out_size,
                         const char*** out_array) {
  API_BEGIN();
  GetDSLAPI()->NodeListAttrNames(
      handle, out_size, out_array);
  API_END();
}
