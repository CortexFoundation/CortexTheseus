/*!
 *  Copyright (c) 2017 by Contributors
 * \file module_util.h
 * \brief Helper utilities for module building
 */
#ifndef CVM_RUNTIME_CVMMODEL_H_
#define CVM_RUNTIME_CVMMODEL_H_

#include <cvm/dlpack.h>
#include <cvm/runtime/packed_func.h>

#include <string>
#include <mutex>

using std::string;

namespace cvm {
namespace runtime {

extern double transpose_int8_avx256_transpose_cnt;
extern double transpose_int8_avx256_gemm_cnt;
extern double im2col_cnt;
extern double cvm_op_rightshift_cnt;
extern double cvm_op_clip_cnt;
extern double cvm_op_dense_cnt;
extern double cvm_op_maxpool_cnt;
extern double cvm_op_broadcast_cnt;
extern double cvm_op_concat_cnt;

struct CVMModel {
public:
  bool loaded{false};
  DLContext ctx;
  CVMModel(const string& graph, DLContext _ctx);
  ~CVMModel();
  int LoadParams(const string& params_str);
  int LoadParamsFromFile(string filepath);
  int GetInputLength();
  int GetOutputLength();
  int64_t GetOps();
  int Run(DLTensor*& input, DLTensor*& output);
  DLTensor* PlanInput();
  DLTensor* PlanInput(char*);
  DLTensor* PlanOutput();
  void SaveTensor(DLTensor* input, char *data);
private:
  int SetInput_(string index, DLTensor* input);
  int Run_();
  int GetOutput_(int index, DLTensor* output);
  PackedFunc set_input;
  PackedFunc get_output;
  PackedFunc load_params;
  PackedFunc get_ops;
  PackedFunc run;
  Module module;
//  std::lock_guard<std::mutex> *lck;
//  static std::mutex mtx;
  int64_t *in_shape{NULL}, *out_shape{NULL};
  int in_ndim, out_ndim;
  int64_t in_size, out_size, model_id;
  int dtype_code{kDLInt};
  int dtype_bits{32};
  int dtype_lanes{1};
};

}
}

#endif // CVM_RUNTIME_CVMMODEL_H_
