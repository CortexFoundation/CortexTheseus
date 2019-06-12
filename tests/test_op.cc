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
#include "npy.hpp"
#include <string.h>
#include <fstream>

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
  uint32_t flatten_data = false;
  std::string attrs;
};

//int ctx = kDLCPU;
int ctx = kDLGPU;

void LoadOp(string op_type, NodeAttrs& attrs) {
  if (op_type == "null") return;
  attrs.name = op_type;
  std::cerr << "op_type " << op_type << "\n";
  attrs.op = cvm::Op::Get(op_type);
  std::cerr << "op_type =====" << op_type << "\n";
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
  arg_ptr->args = args;
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
  int device_type = static_cast<int>(ctx);
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
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, 1, &dl);
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
void test_transpose() {
    string attr_str = " {\"axes\": \"(1, 2, 0)\"} ";
    std::vector<int> dims_ = {3,  3};
    vector<std::vector<int64_t>> shapes_ = {{38, 32, 300},{32, 300, 38}};
    CVMOpParam params;
    params.num_inputs = 1;
    params.num_outputs= 1;
    params.func_name = "transpose";
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, 1, &dl);
      args[i] = *dl;
    }

    std::vector<unsigned long> tshape;
    std::vector<int32_t> tdata;
    npy::LoadArrayFromNumpy("/tmp/transpose/in.x.npy", tshape, tdata);
    int32_t *dldata = static_cast<int32_t*>(args[0].data);
    memcpy(dldata, tdata.data(), sizeof(int32_t) * tdata.size());

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    int32_t *dldata3 = static_cast<int32_t*>(args[1].data);
    std::vector<unsigned long> tshape3;
    std::vector<int32_t> tdata3;
    npy::LoadArrayFromNumpy("/tmp/transpose/out.y.npy", tshape3, tdata3);
    int ret =  memcmp(dldata3, tdata3.data(), sizeof(int32_t) * tdata3.size());
    printf("match %d | %d\n", ret == 0, ret);

    if (true) {
    std::cerr << "Expected\n";
    for (int c = 0; c < shapes_[0][0] *  shapes_[0][1] *  shapes_[0][2] ; c++) {
        std::cerr << tdata3.data()[c] << " ";
    }
    std::cerr << "\n";
    std::cerr << "Got\n";
    for (int c = 0; c < shapes_[1][0] *  shapes_[1][1] *  shapes_[1][2] ; c++) {
        std::cerr << dldata3[c] << " ";
    }
    std::cerr << "\n";
    }
}
void test_take() {
    string attr_str = " {\"axis\": \"0\"} ";
    CVMOpParam params;
    params.func_name = "take";
    params.num_inputs = 2;
    params.num_outputs= 1;
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    std::vector<std::vector<unsigned long>> tshape(args.size());
    std::vector<std::vector<int32_t>> tdata(args.size());
    npy::LoadArrayFromNumpy("/tmp/take/in.x.npy", tshape[0], tdata[0]);
    npy::LoadArrayFromNumpy("/tmp/take/in.w.npy", tshape[1], tdata[1]);
    npy::LoadArrayFromNumpy("/tmp/take/out.y.npy", tshape[2], tdata[2]);
    vector<std::vector<int64_t>> shapes_(args.size());
    std::vector<int> dims_(args.size());
    for (auto idx = 0; idx < args.size(); idx++) {
      shapes_[idx].resize(tshape[idx].size());
      dims_[idx] = (tshape[idx].size());
      std::cout << tshape[idx].size() << "\n";
      for (auto j = 0; j < shapes_[idx].size(); j++) {
        shapes_[idx][j] = tshape[idx][j];
        std::cout << tshape[idx][j] << " ";
      }
      std::cout << "\n";
    }
    DLTensor* cpu_tensor;
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, 1, &dl);
      args[i] = *dl;
      if (i < params.num_inputs) {
        CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
        memcpy(cpu_tensor->data, tdata[i].data(), sizeof(int32_t) * tdata[i].size());
        CVMArrayCopyFromTo(cpu_tensor, dl, nullptr);
        CVMArrayFree(cpu_tensor);
      }
    }

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op_slice = get_func(params, &attr, args, params.num_inputs);
    op_slice();

    vector<int32_t> cpu_output_tensor(tdata[params.num_inputs].size());
    {
      int i = params.num_inputs; // first output
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
      CVMArrayCopyFromTo(&args[i], cpu_tensor, nullptr);
      memcpy(cpu_output_tensor.data(), cpu_tensor->data, sizeof(int32_t) * tdata[i].size());
      CVMArrayFree(cpu_tensor);
    }
    int ret =  memcmp(cpu_output_tensor.data(),
                      tdata[params.num_inputs].data(),
                      sizeof(int32_t) * tdata[params.num_inputs].size());
    printf("match %d | %d\n", ret == 0, ret);

    //if (true) {
    //std::cerr << "Expected " << shapes_[0][0] << " " <<  shapes_[0][1] << " " << shapes_[0][2] << "\n";
    //for (int c = 0; c < tdata[params.num_inputs].size() ; c++) {
    //    std::cerr << tdata[params.num_inputs].data()[c] << " ";
    //}
    //std::cerr << "\n";
    //std::cerr << "Got " << shapes_[0][0] << " " <<  shapes_[0][1] << " " << shapes_[0][2] << "\n";
    //for (int c = 0; c < tdata[params.num_inputs].size() ; c++) {
    //    std::cerr << cpu_output_tensor[c] << " ";
    //}
    //std::cerr << "\n";
    //}
}

void test_op(string op_name, int num_inputs, int num_outputs, int num_test) {
  printf("\ntest %s\n", op_name.c_str());
  for(int i = 0; i < num_test; i++){
    string attr_path = "/tmp/" + op_name + "/attr" + std::to_string(i) + ".txt";
    ifstream infile;
    infile.open(attr_path);
    string attr_str = "";
    getline(infile, attr_str);
    infile.close();
    //string attr_str = " {\"axis\": \"" + std::to_string(i-1) + "\"} ";
    std::cout << attr_str << endl;
    CVMOpParam params;
    params.func_name = op_name;
    params.num_inputs = num_inputs;
    params.num_outputs= num_outputs;
    params.flatten_data = false;
    std::vector<DLTensor> args(params.num_inputs + params.num_outputs);
    std::vector<std::vector<unsigned long>> tshape(args.size());
    std::vector<std::vector<int32_t>> tdata(args.size());
    for(int in_i = 0; in_i < num_inputs; in_i++){
        string in_path = "/tmp/"+op_name+"/in" + std::to_string(i) + std::to_string(in_i) + ".npy";
        cout << in_path << endl;
        npy::LoadArrayFromNumpy(in_path, tshape[in_i], tdata[in_i]);
    }
    string out_path = "/tmp/"+op_name+"/out" + std::to_string(i) + ".npy";
    cout << out_path << endl;
    npy::LoadArrayFromNumpy(out_path, tshape[num_inputs], tdata[num_inputs]);
    vector<std::vector<int64_t>> shapes_(args.size());
    std::vector<int> dims_(args.size());
    for (auto idx = 0; idx < args.size(); idx++) {
      shapes_[idx].resize(tshape[idx].size());
      dims_[idx] = (tshape[idx].size());
      std::cout << "tshape[idx].size() = " << tshape[idx].size() << "\n";
      for (auto j = 0; j < shapes_[idx].size(); j++) {
        shapes_[idx][j] = tshape[idx][j];
        std::cout << tshape[idx][j] << " ";
      }
      std::cout << "\n";
    }
    DLTensor* cpu_tensor;
    for (uint32_t i = 0; i < args.size(); i++) {
      DLTensor* dl;
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, ctx, 1, &dl);
      args[i] = *dl;
      if (i < params.num_inputs) {
        CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
        memcpy(cpu_tensor->data, tdata[i].data(), sizeof(int32_t) * tdata[i].size());
        CVMArrayCopyFromTo(cpu_tensor, dl, nullptr);
        CVMArrayFree(cpu_tensor);
      }
    }

    NodeAttrs attr;
    LoadOp(params.func_name, attr);
    LoadOpAttr(attr_str, attr);
    auto op = get_func(params, &attr, args, params.num_inputs);
    op();

    vector<int32_t> cpu_output_tensor(tdata[params.num_inputs].size());
    {
      int i = params.num_inputs; // first output
      CVMArrayAlloc(shapes_[i].data(), dims_[i], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &cpu_tensor);
      CVMArrayCopyFromTo(&args[i], cpu_tensor, nullptr);
      memcpy(cpu_output_tensor.data(), cpu_tensor->data, sizeof(int32_t) * tdata[i].size());
      CVMArrayFree(cpu_tensor);
    }
    int ret =  memcmp(cpu_output_tensor.data(),
        tdata[params.num_inputs].data(),
        sizeof(int32_t) * tdata[params.num_inputs].size());
    printf("match %d | %d\n", ret == 0, ret);
    assert(ret == 0);
  }
}
int main() {
//    test_take();
//    test_op("concatenate", 2, 1, 4);//pass
//    test_op("repeat", 1, 1, 4); //pass
//    test_op("tile", 1, 1, 5); //pass
//    test_op("transpose", 1, 1, 5);// 5th case failed
//    test_op("strided_slice", 1, 1, 3);
//    test_op("slice_like", 2, 1, 3); // pass
//    test_op("max", 1, 1, 7); // pass
//    test_op("sum", 1,1,7); // pass
//    test_op("take", 2, 1, 2);
    return 0;
}
