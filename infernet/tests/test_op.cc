#include <cvm/c_api.h>
#include <cvm/model.h>
#include <iostream>
#include <thread>
#include <omp.h>
#include <cvm/runtime/registry.h>
#include <cvm/op.h>
#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/serializer.h>
#include <cvm/node.h>
#include <cvm/runtime/c_runtime_api.h>
#include "../npy.hpp"

using namespace std;

using cvm::runtime::PackedFunc;
using cvm::runtime::Registry;
using namespace cvm;
using namespace cvm::runtime;

int dtype_code{kDLInt};
int dtype_bits{32};
int dtype_lanes{1};

struct CVMOpParam {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
  std::string attrs;
};

void LoadOp(string op_type, NodeAttrs& attrs) {
  if (op_type == "null") return;
  attrs.name = op_type;
  attrs.op = cvm::Op::Get(attrs.name);
}
void LoadOpAttr(std::string json_, NodeAttrs& attrs) {
  std::istringstream is(json_);
  utils::JSONReader reader(&is);
  reader.Read(&attrs.dict);
  if (attrs.op->attr_parser) {
    attrs.op->attr_parser(&attrs);
  }
}

struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<CVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
};

std::function<void()> get_func(
    const CVMOpParam& param, NodeAttrs* attr,
    const std::vector<DLTensor>& args,
    size_t num_inputs)
{

  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<CVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };

  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = std::move(args);
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    CVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (param.flatten_data) {
      arg_ptr->shape_data[i] = std::accumulate(
          t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }
  CVMValue t_attr;
  t_attr.v_handle = (void*)attr;
  arg_ptr->arg_values.push_back(t_attr);
  arg_ptr->arg_tcodes.push_back(kHandle);


  auto op = param.func_name;
  int device_type = static_cast<int>(kDLCPU);
  std::string module_name = "cvm.runtime.cvm";
  if (device_type == kDLGPU) module_name += "_cuda";
  module_name += ".";
  auto func = cvm::runtime::Registry::Get(module_name + op);
  VERIFY(func != nullptr) << "function undefined " << module_name + op;
  return [arg_ptr, op, func](){
    CVMRetValue rv;
    CVMArgs targs(
      arg_ptr->arg_values.data(),
      arg_ptr->arg_tcodes.data(),
      static_cast<int>(arg_ptr->arg_values.size())
    );
    func->CallPacked(targs, &rv);
  };

  return [](){};
}
void test_op_take() {
    string attr_str = "{}";
    std::vector<int> dims_ = {4, 1, 4};
    vector<std::vector<int64_t>> shapes_ = {{1, 32, 416, 416}, {1}, {1,32,416, 416}};
    CVMOpParam params;
    params.num_inputs = 2;
    params.num_outputs= 1;
    params.func_name = "broadcast_mul";
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &dl);
      args[i] = *dl;
    }

    std::vector<unsigned long> tshape;
    std::vector<int32_t> tdata;
    npy::LoadArrayFromNumpy("/tmp/yolo/out/broadcast_mul72_0.mrt.dump.in.npy", tshape, tdata);
    int32_t *dldata = static_cast<int32_t*>(args[0].data);
    memcpy(dldata, tdata.data(), sizeof(int32_t) * tdata.size());

    std::vector<unsigned long> tshape2;
    std::vector<int32_t> tdata2;
    int32_t *dldata2 = static_cast<int32_t*>(args[1].data);
    npy::LoadArrayFromNumpy("/tmp/yolo/out/broadcast_mul72_1.mrt.dump.in.npy", tshape2, tdata2);
    memcpy(dldata2, tdata2.data(), sizeof(int32_t) * tdata2.size());

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    int32_t *dldata3 = static_cast<int32_t*>(args[2].data);
    std::vector<unsigned long> tshape3;
    std::vector<int32_t> tdata3;
    npy::LoadArrayFromNumpy("/tmp/yolo/out/broadcast_mul72_0.mrt.dump.out.npy", tshape3, tdata3);
    int ret =  memcmp(dldata3, tdata3.data(), sizeof(int32_t) * tdata3.size());
    printf("%d\n", ret);

}
namespace cvm {
namespace runtime {
extern void matrix_mul(const int8_t *a, const int8_t *b, const int32_t *bias,
        int32_t *c, const int M, const int K, const int N, int algo);
}
}
void test_matrix_mul() {
    int M = 7, K = 2, N = 3;
    vector<int8_t> a(M * K), b(K * N);
    vector<int32_t> c(M * N);
    vector<int32_t> bias(M);
    // std::generate(v.begin(), v.end(), [n = 0] () mutable { return n++; });
    std::generate(a.begin(), a.end(), [n = 0] () mutable { return n++; });
    std::generate(b.begin(), b.end(), [n = 0] () mutable { return n++; });
    std::generate(bias.begin(), bias.end(), [n = 0] () mutable { return n++; });
    matrix_mul(a.data(), b.data(), bias.data(), c.data(), M, K, N, 0);
    for (auto x : c) {
        std::cout << x << " ";
    }
    std::cout << "\n";
}
void test_depthwise_conv () {
    string attr_str = " {\"layout\": \"NCHW\", \"kernel_layout\": \"OIHW\", \"kernel_size\": \"[3, 3]\", \"padding\": \"(1, 1)\", \"use_bias\": \"True\", \"strides\": \"(1, 1)\", \"channels\": \"10\", \"dilation\": \"(1, 1)\", \"groups\": \"1024\"} ";
    std::vector<int> dims_ = {4, 4, 4};
    vector<std::vector<int64_t>> shapes_ = {{1, 1024, 7, 7}, {1024, 1, 3, 3}, {1, 1024, 7, 7}};
    CVMOpParam params;
    params.num_inputs = 2;
    params.num_outputs= 1;
    params.func_name = "conv2d";
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &dl);
      args[i] = *dl;
    }

    std::vector<unsigned long> tshape;
    std::vector<int32_t> tdata;
    npy::LoadArrayFromNumpy("/tmp/conv2d_depthwise/in.x.npy", tshape, tdata);
    int32_t *dldata = static_cast<int32_t*>(args[0].data);
    memcpy(dldata, tdata.data(), sizeof(int32_t) * tdata.size());
    int n_c = shapes_[0][1];
    int i_h = shapes_[0][2];
    int i_w = shapes_[0][3];
    if (false) {
    for (int c = 0; c < shapes_[0][1]; c++) {
      for (int i = 0; i < shapes_[0][2]; i++) {
        for (int j = 0; j < shapes_[0][3]; j++) {
          std::cerr << dldata[(c) * i_h * i_w +  i * i_w + j] << " ";
        }
        std::cerr << "\n";
      }
        std::cerr << "\n";
    }
    }

    std::vector<unsigned long> tshape2;
    std::vector<int32_t> tdata2;
    int32_t *dldata2 = static_cast<int32_t*>(args[1].data);
    npy::LoadArrayFromNumpy("/tmp/conv2d_depthwise/in.w.npy", tshape2, tdata2);
    memcpy(dldata2, tdata2.data(), sizeof(int32_t) * tdata2.size());

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    int32_t *dldata3 = static_cast<int32_t*>(args[2].data);
    std::vector<unsigned long> tshape3;
    std::vector<int32_t> tdata3;
    npy::LoadArrayFromNumpy("/tmp/conv2d_depthwise/out.y.npy", tshape3, tdata3);
    int ret =  memcmp(dldata3, tdata3.data(), sizeof(int32_t) * tdata3.size());
    printf("match %d | %d\n", ret == 0, ret);
    int o_h = shapes_[2][2];
    int o_w = shapes_[2][3];
    if (false) {
    std::cerr << "Expected\n";
    for (int c = 0; c < n_c; c++) {
      for (int i = 0; i < o_h; i++) {
        for (int j = 0; j < o_w; j++) {
          std::cerr << tdata3.data()[(c) * o_h * o_w +  i * o_w + j] << " ";
        }
        std::cerr << "\n";
      }
        std::cerr << "\n";
    }
    std::cerr << "Got\n";
    for (int c = 0; c < n_c; c++) {
      for (int i = 0; i < o_h; i++) {
        for (int j = 0; j < o_w; j++) {
          std::cerr << dldata3[(c ) * o_h * o_w +  i * o_w + j] << " ";
        }
        std::cerr << "\n";
      }
        std::cerr << "\n";
    }
    }
}
int main() {
    test_depthwise_conv();
    return 0;
}
