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
extern double cvm_op_cvm_shift_cnt;
extern double cvm_op_clip_cnt;
extern double cvm_op_dense_cnt;
extern double cvm_op_maxpool_cnt;
extern double cvm_op_broadcast_cnt;
extern double cvm_op_concat_cnt;
extern double cvm_op_upsampling_cnt;
extern double cvm_op_inline_matmul_cnt;
extern double cvm_op_elemwise_cnt;
extern double cvm_op_chnwise_conv_cnt;
extern double cvm_op_chnwise_conv1x1_cnt;
extern double cvm_op_depthwise_conv_cnt;

struct CVMModel {
public:
  bool loaded{false};
  CVMModel(const string& graph, DLContext _ctx);
  ~CVMModel();
  int LoadParams(const string& params_str);
  int LoadParamsFromFile(string filepath);
  int GetInputLength();
  int GetOutputLength();
  int64_t GetStorageSize();
  int64_t GetOps();
  int GetSizeofOutput();
  int Run(DLTensor* input, std::vector<DLTensor*> output);
  DLTensor* PlanInput();
  template<typename Type>
  DLTensor* PlanInput(Type*);
  std::vector<DLTensor*> PlanOutput();
  void SaveTensor(std::vector<DLTensor*> outputs, char *data);
private:
  int SetInput_(string index, DLTensor* input);
  int Run_();
  int GetOutput_(int index, DLTensor* output);
  DLContext ctx_;
  PackedFunc set_input_;
  PackedFunc get_output_;
  PackedFunc load_params_;
  PackedFunc get_ops_;
  PackedFunc run_;
  PackedFunc get_storage_size_;
  Module module_;
  int64_t in_size_;
  int64_t *out_size_;
  int32_t out_num_;
  int64_t model_id_;
  bool is_output_int32_;
  bool is_input_int32_;
  std::vector<int> dims_;
  std::vector<int64_t*> shapes_;
  int dtype_code{kDLInt};
  int dtype_bits{32};
  int dtype_lanes{1};
};

}
}

#endif // CVM_RUNTIME_CVMMODEL_H_
