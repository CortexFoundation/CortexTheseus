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
  out_size_(NULL)
{
//  CVMModel::mtx.lock();
//  lck = new std::lock_guard<std::mutex>(CVMModel::mtx, std::adopt_lock);
  shapes_.clear();
  model_id_ = rand();
  loaded = false;
  ctx = _ctx;
  const PackedFunc* module_creator = Registry::Get("cvm.runtime.create");
  if (module_creator != nullptr) {
    try {
      module_ = (*module_creator)(
        graph,
        static_cast<int>(ctx.device_type),
        static_cast<int>(ctx.device_id)
      );
    } catch (std::exception &e) {
      return;
    }
    auto init = module_.GetFunction("init");
    if (init()) {
      return;
    }
  } else {
    return;
  }
  set_input_ = module_.GetFunction("set_input");
  get_output_ = module_.GetFunction("get_output");
  load_params_ = module_.GetFunction("load_params");
  run_ = module_.GetFunction("run");
  get_ops_ = module_.GetFunction("get_ops");
  get_storage_size_ = module_.GetFunction("get_storage_size");
  auto storage_size = GetStorageSize();

  if (storage_size > (1ll << 38) || storage_size == -1) {
    return;
  }

  auto setup = module_.GetFunction("setup");
  if (setup()) {
    return;
  }
  auto get_output_num = module_.GetFunction("get_output_num");
  get_output_num(&out_num_);

  if (out_num_< 1) {
    return;
  }

  auto get_input_shape = module_.GetFunction("get_input_shape");

  DLTensor* t = new DLTensor();
  t->shape = nullptr;
  get_input_shape("data", t);
  in_size_ = 1;
  for (int i = 0; i < t->ndim; ++i)
    in_size_ *= t->shape[i];

  dims_.push_back(t->ndim);
  int64_t *shape = new int64_t[t->ndim];
  memcpy(shape, t->shape, t->ndim * sizeof(int64_t));
  shapes_.push_back(shape);

  auto get_output_shape = module_.GetFunction("get_output_shape");
  out_size_ = new int64_t[out_num_];
  for (int k = 0; k < out_num_; ++k) {
    out_size_[k] = 1;
    get_output_shape(k, t);
    out_size_[k] = 1;
    for (int i = 0; i < t->ndim; ++i)
        out_size_[k] *= t->shape[i];

    dims_.push_back(t->ndim);
    shape = new int64_t[t->ndim];
    memcpy(shape, t->shape, t->ndim * sizeof(int64_t));
    shapes_.push_back(shape);
 }

  loaded = true;
  delete t->shape;
  delete t;
}

CVMModel::~CVMModel() {
  for (int i = 0; i < shapes_.size(); ++i) {
      if (shapes_[i])
          delete shapes_[i];
  }
  if (out_size_)
      delete out_size_;
//  delete lck;
}

int64_t CVMModel::GetStorageSize() {
  int64_t ret;
  if (get_storage_size_(&ret)) return -1;
  return ret;
}

int64_t CVMModel::GetOps() {
  int64_t ret;
  if (get_ops_(&ret)) return -1;
  return ret;
}

DLTensor* CVMModel::PlanInput() {
  DLTensor* ret;
  CVMArrayAlloc(shapes_[0], dims_[0], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  return ret;
}

DLTensor* CVMModel::PlanInput(char *input) {
  DLTensor* ret = nullptr;
  CVMArrayAlloc(shapes_[0], dims_[0], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  auto data = static_cast<int*>(ret->data);
  for (int i = 0; i < in_size_; ++i) {
    data[i] = input[i];
  }
  return ret;
}

std::vector<DLTensor*> CVMModel::PlanOutput() {
  std::vector<DLTensor*> ret;
  for (int i = 0; i < out_num_; ++i) {
    DLTensor *t;
    CVMArrayAlloc(shapes_[i + 1], dims_[i + 1], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &t);
    ret.push_back(t);
  }
  return ret;
}

void CVMModel::SaveTensor(std::vector<DLTensor*> outputs, char* mem) {
  bool is_same_shape = (outputs.size() > 1);
  for (int k = 1; k < outputs.size(); ++k) {
    if (outputs[k]->ndim != outputs[0]->ndim) {
      is_same_shape = false;
    }
  }
  if (is_same_shape) {
    for (int k = 1; k < outputs.size(); ++k) {
      for (int i = 0; i < outputs[0]->ndim; ++i) {
        if (outputs[k]->shape[i] != outputs[0]->shape[i]) {
          is_same_shape = false;
        }
      }
    }
  }
  if (!is_same_shape) {
    for (int k = 0; k < outputs.size(); ++k) {
      auto data = static_cast<int*>(outputs[k]->data);
      for (int i = 0; i < out_size_[k]; ++i) {
        *mem++ = static_cast<int8_t>(data[i]);
      }
    }
  } else {
    int ndim = outputs[0]->ndim;
    int64_t* shape;
    std::vector<int*> datas;
    for (int k = 0; k < outputs.size(); ++k) {
      datas.push_back(static_cast<int*>(outputs[k]->data));
    }

    std::function<void(int, int)> recur;
    recur = [&](int index, int i){
      index = index * shape[i];
      if (i == ndim - 1) {
        for (int d = 0; d < ndim; ++d) {
          for (int k = 0; k < shape[i]; ++k) {
            *mem++ = static_cast<int8_t>(datas[d][index + k]);
          }
        }
      } else {
        for (int k = 0; k < shape[i]; ++k) {
          recur(index + k, i + 1);
        }
      }
    };
    recur(0, 0);
  }
}

int CVMModel::LoadParams(const string &params) {
  if (params.size() == 0) return -1;
  CVMByteArray arr;
  arr.data = params.c_str();
  arr.size = params.length();
  return load_params_(arr);
}

int CVMModel::SetInput_(string index, DLTensor* input) {
  return input == nullptr ? -1 : set_input_(index, input);
}

int CVMModel::GetOutput_(int index, DLTensor* output) {
  return output == nullptr ? -1 : get_output_(index, output);
}

int CVMModel::Run_() {
  return run_();
}

int CVMModel::Run(DLTensor* input, std::vector<DLTensor*> outputs) {
  int ret = SetInput_("data", input) || Run_();
  if (ret) return ret;

  for (int i = 0; i < outputs.size(); ++i) {
    ret = GetOutput_(i, outputs[i]);
    if (ret) {
      return ret;
    }
  }

  return 0;
}

int CVMModel::GetInputLength() {
  return static_cast<int>(in_size_);
}

int CVMModel::GetOutputLength() {
  int ret = 0;
  for (int i = 0; i < out_num_; ++i)
    ret += static_cast<int>(out_size_[i]);
  return ret;
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
    auto outputs = model->PlanOutput();
    if (input == nullptr) {
      std::cerr << "input == nullptr || output == nullptr" << std::endl;
      ret = -1;
    } else {
      ret = model->Run(input, outputs);
      if (ret == 0) {
        model->SaveTensor(outputs, output_data);
        if (input)
          CVMArrayFree(input);
        for (int i = 0; i < outputs.size(); ++i)
          CVMArrayFree(outputs[i]);
      }
    }
  }
  return ret;
}

