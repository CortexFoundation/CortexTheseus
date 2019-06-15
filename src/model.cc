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
  model_id_ = rand();
  loaded_ = false;
  ctx_ = _ctx;
  const PackedFunc* module_creator = Registry::Get("cvm.runtime.create");
  if (module_creator != nullptr) {
    try {
      module_ = (*module_creator)(
        graph,
        static_cast<int>(ctx_.device_type),
        static_cast<int>(ctx_.device_id)
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
  // std::cerr << "get_output_num = " << out_num_ << "\n";
  if (out_num_< 1) {
    return;
  }
  {
    auto get_postprocess_method = module_.GetFunction("get_postprocess_method");
    char postprocess_s[32];
    get_postprocess_method(postprocess_s);
    postprocess_method_ = std::string(postprocess_s);
    // std::cerr << "postprocess_method_ = " << postprocess_method_ << "\n";
  }

  {
    auto get_input_precision = module_.GetFunction("get_input_precision");
    int input_precision = 0;
    get_input_precision(&input_precision);
    // std::cerr << "input_precision = " << input_precision << "\n";
    input_bytes_ = 1;
    if (input_precision > 8)
      input_bytes_ = 4;
  }
  {
    auto get_output_precision = module_.GetFunction("get_output_precision");
    int output_precision;
    get_output_precision(&output_precision);
    output_bytes_ = 1;
    if (output_precision > 8)
        output_bytes_ = 4;

    if (postprocess_method_ == "argmax") {
      output_bytes_ = 4;
    } else if (postprocess_method_ == "detection") {
      output_bytes_ = 4;
    }
    // std::cerr << " output_precision = " << output_precision << " output_bytes_ = " << (int)output_bytes_ << "\n";
  }
  {
    auto get_version = module_.GetFunction("get_version");
    char version_s[32];
    get_version(version_s);
    version_ = std::string(version_s);
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

  loaded_ = true;
  delete t->shape;
  delete t;
}

CVMModel::~CVMModel() {
  for (size_t i = 0; i < shapes_.size(); ++i) {
      if (shapes_[i])
          delete shapes_[i];
  }
  if (out_size_)
      delete out_size_;
//  delete lck;
}

bool CVMModel::IsReady() const {
  return loaded_;
}

std::string CVMModel::GetVersion() {
  return version_;
}

std::string CVMModel::GetPostprocessMethod() {
  return postprocess_method_;
}

bool CVMModel::SetPostprocessMethod(const string postprocess_method) {
  postprocess_method_ = postprocess_method;
  return true;
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

DLTensor* CVMModel::PlanInput(void *input) {
  DLTensor* ret = nullptr;
  CVMArrayAlloc(shapes_[0], dims_[0], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  auto data = static_cast<int*>(ret->data);
  if (input_bytes_ == 4) {
      for (int i = 0; i < in_size_; ++i) {
          data[i] = static_cast<int32_t*>(input)[i];
          // std::cerr << "data = " << i << " " << data[i] << "\n";
      }
  } else {
      for (int i = 0; i < in_size_; ++i) {
          data[i] = static_cast<int8_t*>(input)[i];
      }
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
  if (postprocess_method_ == "argmax") {
    int32_t* cp = static_cast<int32_t*>((void*)(mem));
    // argmax by dimension -1
    for (size_t k = 0 ; k < (size_t)out_num_; ++k) {
      uint32_t last_dim = shapes_[ input_num_ +  k][dims_[k] - 1];
      uint32_t out_size = out_size_[k];
      uint32_t out_size_ap = out_size / last_dim;
      auto data = static_cast<int*>(outputs[k]->data);
      for (size_t i = 0; i < out_size_ap; i += last_dim) {
        uint32_t max_id = 0;
        for (size_t j = i; j < i + last_dim; j++) {
          if (int8_t(data[j]) > int8_t(data[i + max_id])) {
            // std::cerr << " max " << int(int8_t(data[j])) << " " << int(int8_t(data[i + max_id]))
            //          << " j = " << j - i << "\n";
            max_id = j - i;
          }
        }
        *cp++ = static_cast<int32_t>(max_id);
      }
    }
  } else if (postprocess_method_ == "detection" || outputs.size() > 1) {
    //TODO(tian) FIXME || outputs.size() > 1 is a dirty hack for forward compatbility
    bool can_concat = true;
    if (outputs.size() == 1) {
      can_concat = false;
    }
    std::vector<uint64_t> xs(outputs.size());
    std::vector<uint64_t> ys(outputs.size());
    if (can_concat) {
      for (size_t k = 0; k < outputs.size(); ++k) {
        ys[k] = outputs[k]->shape[outputs[k]->ndim - 1];
        xs[k] = out_size_[k] / ys[k];
      }
      for (size_t k = 1; k < outputs.size(); ++k) {
        if (xs[k] != xs[0]) {
          can_concat = false;
          break;
        }
      }
    }
    if (can_concat) {
      if (output_bytes_ == 4) {
        int32_t* cp = static_cast<int32_t*>((void*)(mem));
        int32_t nx = xs[0];  // truncate result
        for (int xidx = 0; xidx < nx; xidx++) {
          //   std::cerr << "int8 \n";
          // for (size_t k = 0; k < outputs.size(); ++k) {
          //   auto data = static_cast<uint8_t*>(outputs[k]->data) + xidx * ys[k] * 4;
          //   for (size_t i = 0; i < ys[k] * 4; ++i) {
          //     std::cerr << (int)((uint8_t)data[i]) << " " ;
          //   }
          // }
          //   std::cerr << "\n";

          // std::cerr << "int32 \n";
          for (size_t k = 0; k < outputs.size(); ++k) {
            auto data = static_cast<int32_t*>(outputs[k]->data) + xidx * ys[k];
            for (size_t i = 0; i < ys[k]; ++i) {
              *cp = data[i];
          //    std::cerr << *cp << " " ;
              ++cp;
            }
          }
          //   std::cerr << "\n";
        }
      } else {
        auto cp = mem;
        for (size_t xidx = 0; xidx < xs[0]; xidx++) {
          for (size_t k = 0; k < outputs.size(); ++k) {
            auto data = static_cast<int*>(outputs[k]->data) + xidx * ys[k];
            for (size_t i = 0; i < ys[k]; ++i) {
              *cp++ = static_cast<int8_t>(data[i]);
            }
          }
        }
      }
    } else {
      std::cerr << "yolo post process failed\n";
    }
  } else {
      for (size_t k = 0; k < outputs.size(); ++k) {
        auto data = static_cast<int*>(outputs[k]->data);
        for (int i = 0; i < out_size_[k]; ++i) {
          *mem++ = static_cast<int8_t>(data[i]);
        }
      }
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

  for (size_t i = 0; i < outputs.size(); ++i) {
    ret = GetOutput_(i, outputs[i]);
    if (ret) {
      return ret;
    }
  }

  return 0;
}

int CVMModel::GetInputLength() {
  // std::cerr << " GetInputLength = " << (int)in_size_ << " " << (int)input_bytes_ << "\n";
  return static_cast<int>(in_size_) * input_bytes_;
}

int CVMModel::GetOutputLength() {
  if (postprocess_method_ == "argmax") {
    // argmax by dimension -1
    int ret = 0;
    for (size_t k = input_num_; k < (size_t)input_num_ + out_num_; ++k) {
      uint32_t last_dim = shapes_[k][dims_[k] - 1];
      uint32_t out_size = out_size_[k - input_num_];
      uint32_t out_size_ap = out_size / last_dim;
      // std::cerr << "output[" << k << "]" << "last_dim = " << last_dim << "out_size_[k] = " << out_size << "\n";
      ret += out_size_ap;
    }
    ret *= output_bytes_;
    // std::cerr << "ret = " << ret << "\n";
    return ret;
  }
  else if (postprocess_method_ == "detection") {
    int ret = 0;
    int yolo_num_ret = dims_[input_num_] >= 2 ? shapes_[input_num_][dims_[input_num_] - 2]: 0;
    // std::cerr << "yolo_num_ret = " << yolo_num_ret << "\n";
    for (size_t k = input_num_; k < (size_t)input_num_ + out_num_; ++k) {
      uint32_t last_dim = shapes_[k][dims_[k] - 1];
      // std::cerr << "output[" << k << "] " << "last_dim = " << last_dim  << "\n";
      ret += last_dim;
    }
    ret *= yolo_num_ret;
    ret *= output_bytes_;
    // std::cerr << "output length = " << ret << "\n";
    return ret;
  } else {
    int ret = 0;
    for (int i = 0; i < out_num_; ++i)
      ret += static_cast<int>(out_size_[i]);
    ret *= output_bytes_;
    return ret;
  }
}

int CVMModel::GetSizeOfOutputType() {
  return output_bytes_;
}

int CVMModel::GetSizeOfInputType() {
  return input_bytes_;
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

void* CVMAPILoadModel(const char *graph_fname,
                      const char *model_fname,
                      int device_type,
                      int device_id)
{
  // std::cerr << "graph_fname = " << graph_fname
  //           << "\nmodel_fname = " << model_fname
  //           << "\ndevice_type = " << device_type
  //           << "\ndevice_id = " << device_id << "\n";
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
  if (!model->IsReady() || model->LoadParams(params)) {
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

void CVMAPIGetVersion(void *model_, char* version) {
  CVMModel* model = (CVMModel*)model_;
  if (model == nullptr) return;
  strcpy(version, model->GetVersion().c_str());
}

void CVMAPIGetPreprocessMethod(void *model_, char* method) {
  CVMModel* model = (CVMModel*)model_;
  if (model == nullptr) return;
  strcpy(method, model->GetPostprocessMethod().c_str());
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

int CVMAPISizeOfOutputType(void *model_) {
  CVMModel* model = (CVMModel*)model_;
  if (model != nullptr) {
    return model->GetSizeOfOutputType();
  }
  return 0;
}

int CVMAPISizeOfInputType(void *model_) {
  CVMModel* model = (CVMModel*)model_;
  if (model != nullptr) {
    return model->GetSizeOfInputType();
  }
  return 0;
}

int CVMAPIInfer(void* model_, char *input_data, char *output_data) {
  int ret = 0;
  try {
    if (input_data == nullptr) {
      std::cerr << "input_data error" << std::endl;
      ret = -1;
    } else if (output_data == nullptr) {
      std::cerr << "output error" << std::endl;
      ret = -1;
    } else {
      CVMModel* model = (CVMModel*)model_;
      DLTensor* input = nullptr;
      input = model->PlanInput(input_data);
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
          for (size_t i = 0; i < outputs.size(); ++i)
            CVMArrayFree(outputs[i]);
        }
      }
    }
  } catch (std::exception &e) {
    return -1;
  }
  return ret;
}

