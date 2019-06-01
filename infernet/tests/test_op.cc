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
    string attr_str = "{\"end\": \"[16, 169, 3, 25]\", \"begin\": \"[0, 0, 0, 5]\"}";
    std::vector<int> dims_ = {4, 4};
    vector<std::vector<int64_t>> shapes_ = {{16, 169, 3, 25}, {16, 169, 3, 20}};
    CVMOpParam params;
    params.num_inputs = 1;
    params.num_outputs= 1;
    params.func_name = "strided_slice";
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &dl);
      args[i] = *dl;
    }

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();
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
int main() {
    test_matrix_mul();
    return 0;
}
