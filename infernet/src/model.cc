#include <cvm/c_api.h>
#include <cvm/model.h>
#include <cvm/dlpack.h>
#include <cvm/runtime/module.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <time.h>

using std::string;

namespace cvm {
namespace runtime {

std::mutex CVMModel::mtx;

CVMModel::CVMModel(const string& graph, DLContext _ctx):
  in_shape(NULL), out_shape(NULL)
{
  CVMModel::mtx.lock();
  lck = new std::lock_guard<std::mutex>(CVMModel::mtx, std::adopt_lock);
  model_id = rand();
  loaded = false;
  ctx = _ctx;
  printf("Model #%d was created.\n", model_id);
  const PackedFunc* module_creator = Registry::Get("cvm.runtime.create");
  if (module_creator != nullptr) {
    try {
      module = (*module_creator)(
        graph,
        static_cast<int>(ctx.device_type),
        static_cast<int>(ctx.device_id)
      );
    } catch (std::exception &e) {
      return;
    }
    auto setup = module.GetFunction("setup");
    if (setup()) {
      return;
    }
    loaded = true;
  } else {
    return;
  }
  set_input = module.GetFunction("set_input");
  get_output = module.GetFunction("get_output");
  load_params = module.GetFunction("load_params");
  run = module.GetFunction("run");
  auto get_input_shape = module.GetFunction("get_input_shape");
  DLTensor* t = new DLTensor();
  get_input_shape("data", t);
  in_ndim = t->ndim;
  in_shape = new int64_t[in_ndim];
  memcpy(in_shape, t->shape, in_ndim * sizeof(int64_t));
  in_size = 1;
  for (int i = 0; i < in_ndim; ++i) in_size *= in_shape[i];

  auto get_output_shape = module.GetFunction("get_output_shape");
  get_output_shape(0, t);
  out_ndim = t->ndim;
  out_shape = new int64_t[out_ndim];
  memcpy(out_shape, t->shape, out_ndim * sizeof(int64_t));
  out_size = 1;
  for (int i = 0; i < out_ndim; ++i) out_size *= out_shape[i];

  delete t->shape;
  delete t;
}

CVMModel::~CVMModel() {
  printf("Model #%d was destoried.\n", model_id);
  if (in_shape) delete in_shape;
  if (out_shape) delete out_shape;
  delete lck;
}

DLTensor* CVMModel::PlanInput() {
  DLTensor* ret;
  CVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  return ret;
}

DLTensor* CVMModel::PlanInput(char *input) {
  DLTensor* ret = nullptr;
  CVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  auto data = static_cast<int*>(ret->data);
  for (int i = 0; i < in_size; ++i) {
    data[i] = input[i];
  }
  return ret;
}

DLTensor* CVMModel::PlanOutput() {
  DLTensor* ret;
  CVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  return ret;
}

void CVMModel::SaveTensor(DLTensor* input, char* mem) {
  auto data = static_cast<int*>(input->data);
  for (int i = 0; i < out_size; ++i) {
    mem[i] = static_cast<int8_t>(data[i]);
  }
}

int CVMModel::LoadParams(const string &params) {
  if (params.size() == 0) return -1;
  CVMByteArray arr;
  arr.data = params.c_str();
  arr.size = params.length();
  return load_params(arr);
}

int CVMModel::SetInput_(string index, DLTensor* input) {
  return input == nullptr ? -1 : set_input(index, input);
}

int CVMModel::GetOutput_(int index, DLTensor* output) {
  return output == nullptr ? -1 : get_output(index, output);
}

int CVMModel::Run_() {
  return run();
}

int CVMModel::Run(DLTensor*& input, DLTensor*& output) {
  int ret = SetInput_("data", input) ||
    Run_() ||
    GetOutput_(0, output);
  return ret;
}

int CVMModel::GetInputLength() {
  return static_cast<int>(in_size);
}

int CVMModel::GetOutputLength() {
  return static_cast<int>(out_size);
}

int CVMModel::LoadParamsFromFile(string filepath) {
  std::ifstream input_stream(filepath, std::ios::binary);
  std::string params = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
  input_stream.close();
  return LoadParams(params);
}

/*
int estimate_ops() {
  std::ifstream json_in("/tmp/mnist.nnvm.compile.json", std::ios::in);
  string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();
  auto f = Registry::Get("cvm.runtime.estimate_ops");
  if (f == nullptr) return;
  int ret;
  ret = (*f)(json_data);
}
*/

}
}

string LoadFromFile(string filepath) {
  std::ifstream input_stream(filepath, std::ios::in);
  string str = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
  input_stream.close();
  return str;
}

string LoadFromBinary(string filepath) {
  std::ifstream input_stream(filepath, std::ios::binary);
  string str = string((std::istreambuf_iterator<char>(input_stream)), std::istreambuf_iterator<char>());
  input_stream.close();
  return str;
}

using cvm::runtime::CVMModel;

void* CVMAPILoadModel(const char *graph_fname, const char *model_fname) {
  int id = rand();
  printf("CVMAPI Load Model %d Start\n", id);
  string graph = LoadFromFile(string(graph_fname));
  CVMModel* model = new CVMModel(graph, DLContext{kDLGPU, 1});
  string params = LoadFromBinary(string(model_fname));
  if (!model->loaded || model->LoadParams(params)) {
    delete model;
    return NULL;
  }
  printf("CVMAPI Load Model %d Stop\n", id);
  return (void*)model;
}

void CVMAPIFreeModel(void* model_) {
  int id = rand();
  printf("CVMAPI Free Model %d Start %#llx\n", id, (int64_t)model_);
  CVMModel* model = static_cast<CVMModel*>(model_);
  if (model_) delete model; 
  printf("CVMAPI Free Model %d Stop\n", id);
}

int CVMAPIGetInputLength(void* model_) {
  CVMModel* model = (CVMModel*)model_;
  int ret = model->GetInputLength();
  return ret;
}

int CVMAPIGetOutputLength(void* model_) {
  CVMModel* model = (CVMModel*)model_;
  if (model == nullptr)
      return 0;
  int ret = model->GetOutputLength();
  return ret;
}

int CVMAPIInfer(void* model_, char *input_data, char *output_data) {
  int id = rand(), ret = 0;
  printf("CVMAPI Infer Model %d Start, in %#llx out %#llx\n", id,
      (int64_t)(input_data),
      (int64_t)(output_data)
    );
  if (input_data == nullptr) {
    std::cerr << "input_data error" << std::endl;
    ret = -1;
  } else if (output_data == nullptr) {
    std::cerr << "output error" << std::endl;
    ret = -1;
  } else {
    CVMModel* model = (CVMModel*)model_;
    DLTensor* input = model->PlanInput(input_data);
    DLTensor* output = model->PlanOutput();
    if (input == nullptr || output == nullptr) {
      std::cerr << "input == nullptr || output == nullptr" << std::endl;
      ret = -1;
    } else {
      ret = model->Run(input, output);
      if (ret == 0) {
        model->SaveTensor(output, output_data);
        if (input)
          CVMArrayFree(input);
        if (output)
          CVMArrayFree(output);
      }
    }
  }
  printf("CVMAPI Infer Model %d End\n", id);
  return ret;
}

