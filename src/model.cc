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

// std::mutex CVMModel::mtx;

CVMModel::CVMModel(const string& graph, DLContext _ctx):
  out_size(NULL)
{
//  CVMModel::mtx.lock();
//  lck = new std::lock_guard<std::mutex>(CVMModel::mtx, std::adopt_lock);
  model_id = rand();
  loaded = false;
  ctx = _ctx;
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
    auto init = module.GetFunction("init");
    if (init()) {
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
  get_ops = module.GetFunction("get_ops");
  get_storage_size = module.GetFunction("get_storage_size");
  auto get_output_num = module.GetFunction("get_output_num");
  out_num = get_output_num();

  auto get_input_shape = module.GetFunction("get_input_shape");
  DLTensor* t = new DLTensor();
  t->shape = nullptr;
  get_input_shape("data", t);
  in_size = 1;
  for (int i = 0; i < t->ndim; ++i) in_size *= t->shape[i];
  
  dims.push_back(t->ndim);
  int64_t *shape = new int64_t[t->ndim];
  memcpy(shape, t->shape, t->ndim * sizeof(int64_t));
  shapes.push_back(shape);

  auto get_output_shape = module.GetFunction("get_output_shape");
  out_size = new int64_t[out_num];
  for (int k = 0; k < out_num; ++k) {
    out_size[k] = 1;
    get_output_shape(k, t);
    out_size[k] = 1;
    for (int i = 0; i < t->ndim; ++i) out_size[k] *= t->shape[i]; 
    
    dims.push_back(t->ndim);
    shape = new int64_t[t->ndim];
    memcpy(shape, t->shape, t->ndim * sizeof(int64_t));
    shapes.push_back(shape);
 }

  delete t->shape;
  delete t;
}

CVMModel::~CVMModel() {
  for (int i = 0; i < shapes.size(); ++i) {
    delete shapes[i];
  }
  if (out_size) delete out_size;
//  delete lck;
}

int64_t CVMModel::GetStorageSize() {
  int64_t ret;
  if (get_storage_size(&ret)) return -1;
  return ret;
}

int64_t CVMModel::GetOps() {
  int64_t ret;
  if (get_ops(&ret)) return -1;
  return ret;
}

DLTensor* CVMModel::PlanInput() {
  DLTensor* ret;
  CVMArrayAlloc(shapes[0], dims[0], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  return ret;
}

DLTensor* CVMModel::PlanInput(char *input) {
  DLTensor* ret = nullptr;
  CVMArrayAlloc(shapes[0], dims[0], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  auto data = static_cast<int*>(ret->data);
  for (int i = 0; i < in_size; ++i) {
    data[i] = input[i];
  }
  return ret;
}

std::vector<DLTensor*> CVMModel::PlanOutput() {
  vector<DLTensor*> ret;
  for (int i = 0; i < out_num; ++i) {
    DLTensor *t;
    CVMArrayAlloc(shapes[i + 1], dims[i + 1], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &t);
    ret.push_back(t);
  }
  return ret;
}

void CVMModel::SaveTensor(DLTensor* output, char* mem) {
  auto data = static_cast<int*>(output->data);
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

int CVMModel::Run(DLTensor* input, vector<DLTensor*> output) {
  int ret = SetInput_("data", input) ||
    Run_();

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

void* CVMAPILoadModel(const char *graph_fname, const char *model_fname,
        int device_type, int device_id) {
  string graph, params;
  try {
    graph = LoadFromFile(string(graph_fname));
  } catch (std::exception &e) {
    return NULL;
  }
  CVMModel* model = nullptr;
  if (device_type == 0) {
    model = new CVMModel(graph, DLContext{kDLCPU, 0});
  } else {
    model = new CVMModel(graph, DLContext{kDLGPU, device_id});
  }
  try {
    params = LoadFromBinary(string(model_fname));
  } catch (std::exception &e) {
    return NULL;
  }
  if (!model->loaded || model->LoadParams(params)) {
    delete model;
    return NULL;
  }
  return (void*)model;
}

void CVMAPIFreeModel(void* model_) {
  CVMModel* model = static_cast<CVMModel*>(model_);
  if (model_) delete model;
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

long long CVMAPIGetGasFromModel(void *model_) {
  CVMModel* model = (CVMModel*)model_;
  long long ret = -1;
  if (model != nullptr) {
    ret = static_cast<long long>(model->GetOps());
  }
  return ret;
}

long long CVMAPIGetGasFromGraphFile(char *graph_fname) {
  string json_data;
  try {
    std::ifstream json_in(string(graph_fname), std::ios::in);
    json_data = string((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();
  } catch (std::exception &e) {
    return -1;
  }
  auto f = cvm::runtime::Registry::Get("cvm.runtime.estimate_ops");
  if (f == nullptr) return -1;
  int64_t ret;
  ret = (*f)(json_data);
  return static_cast<long long>(ret);
}

long long CVMAPIGetStorageSize(void *model_) {
  CVMModel* model = (CVMModel*)model_;
  long long ret = -1;
  if (model != nullptr) {
    ret = static_cast<long long>(model->GetStorageSize());
  }
  return ret;
}

int CVMAPIInfer(void* model_, char *input_data, char *output_data) {
  int ret = 0;
  if (input_data == nullptr) {
    std::cerr << "input_data error" << std::endl;
    ret = -1;
  } else if (output_data == nullptr) {
    std::cerr << "output error" << std::endl;
    ret = -1;
  } else {
    CVMModel* model = (CVMModel*)model_;
    DLTensor* input = model->PlanInput(input_data);
    auto output = model->PlanOutput();
    if (input == nullptr) {
      std::cerr << "input == nullptr || output == nullptr" << std::endl;
      ret = -1;
    } else {
      ret = model->Run(input, output);
      if (ret == 0) {
        model->SaveTensor(output, output_data);
        if (input)
          CVMArrayFree(input);
        for (int i = 0; i < output.size(); ++i)
          CVMArrayFree(output[i]);
      }
    }
  }
  return ret;
}

