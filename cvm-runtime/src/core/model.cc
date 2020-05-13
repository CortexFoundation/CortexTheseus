#include <cvm/c_api.h>
#include <cvm/model.h>
#include <cvm/dlpack.h>
#include <cvm/errors.h>
#include <cvm/runtime/module.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <exception>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <time.h>

using std::string;

namespace cvm {
namespace runtime {

CVMModel::CVMModel(const string& graph, DLContext _ctx):
  out_size_(nullptr), loaded_(false)
{
  ctx_ = _ctx;
  const PackedFunc* module_creator = Registry::Get("cvm.runtime.create");
  module_ = (*module_creator)(graph,
                              static_cast<int>(ctx_.device_type),
                              static_cast<int>(ctx_.device_id));

  set_input_ = module_.GetFunction("set_input");
  get_output_ = module_.GetFunction("get_output");
  load_params_ = module_.GetFunction("load_params");
  run_ = module_.GetFunction("run");
  get_ops_ = module_.GetFunction("get_ops");
  get_storage_size_ = module_.GetFunction("get_storage_size");
  GetStorageSize();

  auto get_output_num = module_.GetFunction("get_output_num");
  get_output_num(&out_num_);
  if (out_num_< 1) {
    return;
  }
  {
    auto get_postprocess_method = module_.GetFunction("get_postprocess_method");
    char postprocess_s[32];
    get_postprocess_method(postprocess_s);
    postprocess_method_ = std::string(postprocess_s);
  }

  {
    auto get_input_precision = module_.GetFunction("get_input_precision");
    int input_precision = 0;
    get_input_precision(&input_precision);
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
      if (shapes_[i]){
          delete shapes_[i];
      }
  }
  if (out_size_){
      delete out_size_;
  }
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
  get_storage_size_(&ret);
  return ret;
}

int64_t CVMModel::GetOps() {
  int64_t ret;
  get_ops_(&ret);
  return ret;
}

DLTensor* CVMModel::PlanInput(void *input, int size) {
  VERIFY_EQ(this->GetInputLength(), size);
  DLTensor* ret = nullptr;
  CVMArrayAlloc(shapes_[0], dims_[0], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &ret);
  auto data = static_cast<int32_t*>(ret->data);
  if (input_bytes_ == 4) {
    for (int i = 0; i < in_size_; ++i) {
      data[i] = static_cast<int32_t*>(input)[i];
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
    CVMArrayAlloc(shapes_[i + input_num_], dims_[i + input_num_], dtype_code, dtype_bits, dtype_lanes, kDLCPU, 0, &t);
    ret.push_back(t);
  }
  return ret;
}

void CVMModel::SaveTensor(std::vector<DLTensor*> outputs, char* mem) {
  if (postprocess_method_ == "argmax") {
    int32_t* cp = static_cast<int32_t*>((void*)(mem));
    // argmax by dimension -1
    for (size_t k = 0 ; k < (size_t)out_num_; ++k) {
      uint32_t last_dim = shapes_[ input_num_ +  k][dims_[input_num_ + k] - 1];
      uint32_t out_size = out_size_[k];
      uint32_t out_size_ap = out_size / last_dim;
      auto data = static_cast<int*>(outputs[k]->data);
      for (size_t i = 0; i < out_size_ap; i += last_dim) {
        uint32_t max_id = 0;
        for (size_t j = i; j < i + last_dim; j++) {
          if (int8_t(data[j]) > int8_t(data[i + max_id])) {
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
          for (size_t k = 0; k < outputs.size(); ++k) {
            auto data = static_cast<int32_t*>(outputs[k]->data) + xidx * ys[k];
            for (size_t i = 0; i < ys[k]; ++i) {
              *cp = data[i];
              ++cp;
            }
          }
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
      // TODO(wentao): verify in model load.
      VERIFY(false) << "detection model output cannot concat";
    }
  } else if (output_bytes_ == 4){
    int32_t* ref = static_cast<int32_t*>((void*)mem);
    for (size_t k = 0; k < outputs.size(); ++k) {
      int32_t const* data = static_cast<int32_t*>(outputs[k]->data);
      memcpy(ref, data, sizeof(int32_t) * out_size_[k]);
      ref += out_size_[k];
    }
  } else {
    for (size_t k = 0; k < outputs.size(); ++k) {
      int32_t const* data = static_cast<int32_t*>(outputs[k]->data);
      for (int i = 0; i < out_size_[k]; ++i) {
        *mem++ = static_cast<int8_t>(data[i]);
      }
    }
  }
}

int CVMModel::LoadParams(const string &params) {
  VERIFY_NE(params.size(), 0);
  CVMByteArray arr;
  arr.data = params.c_str();
  arr.size = params.length();
  load_params_(arr);
  return SUCCEED;
}

void CVMModel::SetInput_(string index, DLTensor* input) {
  CHECK(input != nullptr);
  set_input_(index, input);
}

void CVMModel::GetOutput_(int index, DLTensor* output) {
  CHECK(output != nullptr);
  get_output_(index, output);
}

void CVMModel::Run(DLTensor* input, std::vector<DLTensor*> outputs) {
  SetInput_("data", input);
  run_();

  for (size_t i = 0; i < outputs.size(); ++i) {
    GetOutput_(i, outputs[i]);
  }
}

int CVMModel::GetInputLength() {
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
      ret += out_size_ap;
    }
    ret *= output_bytes_;
    return ret;
  }
  else if (postprocess_method_ == "detection") {
    int ret = 0;
    int yolo_num_ret = dims_[input_num_] >= 2 ? shapes_[input_num_][dims_[input_num_] - 2]: 0;
    for (size_t k = input_num_; k < (size_t)input_num_ + out_num_; ++k) {
      uint32_t last_dim = shapes_[k][dims_[k] - 1];
      ret += last_dim;
    }
    ret *= yolo_num_ret;
    ret *= output_bytes_;
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

}
}
