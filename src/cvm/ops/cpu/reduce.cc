#include "ops.h"

namespace cvm {
namespace runtime {

inline std::vector<int64_t> GetRealAxis(TShape& axis, bool exclude, DLTensor *x){
  for(size_t i = 0; i < axis.ndim(); i++){
    if(axis[i] < 0) axis[i] += x->ndim;
    //VERIFY(axis[i] >= 0 && axis[i] < x->ndim);
  }
  std::vector<int64_t> raxis;
  try{
    if(!exclude){
      for(size_t i = 0; i < axis.ndim(); i++){
        raxis.push_back(axis[i]);
      }
    }else{
      raxis.resize(x->ndim - axis.ndim());
      for(int i = 0, k = 0; i < x->ndim; i++){
        bool flag = false;
        for(size_t j = 0; j < axis.ndim(); j++){
          if(axis[j] == i) {
            flag = true;
            break;
          }
        }
        if(!flag){
          raxis[k++] = i;
        }
      }
    }
  }catch(const std::bad_alloc& e){
    CHECK(false) << e.what();
  }
  return raxis;
}

template<typename F>
inline void Reduce(DLTensor *x, DLTensor *y, TShape& axis, bool exclude, F const &f){
  int32_t *x_data = static_cast<int32_t*>(x->data);
  int32_t *y_data = static_cast<int32_t*>(y->data);

  std::vector<int64_t> realAxis = GetRealAxis(axis, exclude, x);

  if(exclude && realAxis.size() == 0){
    memcpy(y_data, x_data, getSize(x) * sizeof(int32_t));
  }
  else if(realAxis.size() == 0){
    int32_t tmp = 0;
    for(uint64_t i = 0; i < getSize(x); i++){
      //sum += x_data[i];
      f(tmp, x_data[i]);
    }
    y_data[0] = tmp;
  }else{
    try{
      std::vector<bool> flag(x->ndim, false);
      for(uint32_t i = 0; i < realAxis.size(); i++){
        int32_t val = realAxis[i];
        flag[val] = true;
      }
      std::sort(realAxis.begin(), realAxis.end());
      realAxis.resize(std::unique(realAxis.begin(), realAxis.end()) - realAxis.begin());

      uint64_t axis_size = 1;
      for(uint32_t i = 0; i < realAxis.size(); i++){
        axis_size *= x->shape[realAxis[i]];
      }
      std::vector<uint64_t> every_xdim_size(x->ndim, 1);
      for(int i = x->ndim-2; i >= 0; i--){
        every_xdim_size[i] = x->shape[i+1] * every_xdim_size[i+1];
      }
      int32_t yndim = y->ndim;
      std::vector<int64_t> yshape(y->ndim);
      for(int i = 0; i < y->ndim; i++){
        yshape[i] = y->shape[i];
      }
      for(int i = 0, j = 0; i < y->ndim; i++){
        if(y->shape[i] == 1) {
          yndim -= 1;
        }else{
          yshape[j++] = y->shape[i];
        }
      }
      for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t in_i = 0, o_i = i;
        for(int j = yndim-1, xj = x->ndim-1; j>=0; j--){
          uint64_t col = o_i % yshape[j];
          o_i /= yshape[j];
          while(xj >= 0 && flag[xj--]);
          in_i += col * every_xdim_size[xj+1];
        }
        int32_t tmp = x_data[in_i];
        for(uint64_t xi = 1; xi < axis_size; xi++){
          uint64_t o_i = xi, tmp_in_i = 0;
          for(int j = realAxis.size() - 1; j>=0; j--){
            uint64_t col = o_i % x->shape[realAxis[j]];
            o_i /= x->shape[realAxis[j]];
            tmp_in_i += col * every_xdim_size[realAxis[j]];
          }
          f(tmp, x_data[in_i+tmp_in_i]);
        }
        y_data[i] = tmp;
      }
    }catch(const std::bad_alloc& e){
      CHECK(false) << e.what();
    }
  }
}

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.sum")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
  TShape axis = param.axis;
  //bool keepdims = param.keepdims; //the reduce axis is always 1
  bool exclude = param.exclude;

  auto f = [&](int32_t& tmp, int32_t value)->void {
    tmp += value;
  };

  Reduce(x, y, axis, exclude, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.max")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
  TShape axis = param.axis;
  //bool keepdims = param.keepdims; //the reduce axis is always 1
  bool exclude = param.exclude;

  auto f = [&](int32_t& tmp, int32_t value)->void {
    if(tmp < value) tmp = value;
  };

  Reduce(x, y, axis, exclude, f);
});

}
}
