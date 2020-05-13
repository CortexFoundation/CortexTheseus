/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_runtime_api.cc
 * \brief Device specific implementations
 */
#include <cvm/errors.h>
#include <cvm/runtime/c_runtime_api.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>

using namespace cvm::runtime;

int CVMModGetFunction(CVMModuleHandle mod,
                      const char* func_name,
                      int query_imports,
                      CVMFunctionHandle *func) {
  API_BEGIN();
  PackedFunc pf = static_cast<Module*>(mod)->GetFunction(func_name);
  if (pf != nullptr) {
    *func = new PackedFunc(pf);
  } else {
    *func = nullptr;
  }
  API_END();
}

int CVMModFree(CVMModuleHandle mod) {
  API_BEGIN();
  delete static_cast<Module*>(mod);
  API_END();
}

int CVMFuncFree(CVMFunctionHandle func) {
  API_BEGIN();
  delete static_cast<PackedFunc*>(func);
  API_END();
}

int CVMFuncCall(CVMFunctionHandle func,
                CVMValue* args,
                int* arg_type_codes,
                int num_args,
                CVMValue* ret_val,
                int* ret_type_code) {
  API_BEGIN();
  CVMRetValue rv;
  (*static_cast<const PackedFunc*>(func)).CallPacked(
      CVMArgs(args, arg_type_codes, num_args), &rv);
  // handle return string.
  if (rv.type_code() == kStr ||
     rv.type_code() == kCVMType ||
      rv.type_code() == kBytes) {
    CVMRuntimeEntry* e = CVMAPIRuntimeStore::Get();
    if (rv.type_code() != kCVMType) {
      e->ret_str = *rv.ptr<std::string>();
    } else {
      e->ret_str = rv.operator std::string();
    }
    if (rv.type_code() == kBytes) {
      e->ret_bytes.data = e->ret_str.c_str();
      e->ret_bytes.size = e->ret_str.length();
      *ret_type_code = kBytes;
      ret_val->v_handle = &(e->ret_bytes);
    } else {
      *ret_type_code = kStr;
      ret_val->v_str = e->ret_str.c_str();
    }
  } else {
    rv.MoveToCHost(ret_val, ret_type_code);
  }
  API_END();
}

int CVMCFuncSetReturn(CVMRetValueHandle ret,
                      CVMValue* value,
                      int* type_code,
                      int num_ret) {
  API_BEGIN();
  CHECK_EQ(num_ret, 1);
  CVMRetValue* rv = static_cast<CVMRetValue*>(ret);
  *rv = CVMArgValue(value[0], type_code[0]);
  API_END();
}

int CVMFuncCreateFromCFunc(CVMPackedCFunc func,
                           void* resource_handle,
                           CVMPackedCFuncFinalizer fin,
                           CVMFunctionHandle *out) {
  API_BEGIN();
  if (fin == nullptr) {
    *out = new PackedFunc(
        [func, resource_handle](CVMArgs args, CVMRetValue* rv) {
          int ret = func((CVMValue*)args.values, (int*)args.type_codes, // NOLINT(*)
                         args.num_args, rv, resource_handle);
          if (ret != 0) {
            throw utils::Error(CVMGetLastError() + ::utils::StackTrace());
          }
        });
  } else {
    // wrap it in a shared_ptr, with fin as deleter.
    // so fin will be called when the lambda went out of scope.
    std::shared_ptr<void> rpack(resource_handle, fin);
    *out = new PackedFunc(
        [func, rpack](CVMArgs args, CVMRetValue* rv) {
          int ret = func((CVMValue*)args.values, (int*)args.type_codes, // NOLINT(*)
                         args.num_args, rv, rpack.get());
          if (ret != 0) {
            throw utils::Error(CVMGetLastError() + ::utils::StackTrace());
          }
      });
  }
  API_END();
}
