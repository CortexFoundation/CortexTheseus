#include "ops.h"

namespace cvm {
namespace runtime {
  
typedef std::function<int32_t(int32_t a, int32_t b)> elemwise_func;

inline void elemwise(DLTensor *args0, DLTensor *args1, DLTensor *args2, const elemwise_func& f){
  int32_t *a = static_cast<int32_t*>(args0->data);
  int32_t *b = static_cast<int32_t*>(args1->data);
  int32_t *c = static_cast<int32_t*>(args2->data);

#pragma omp parallel for
  for(uint64_t i = 0; i < getSize(args0); i++){
    c[i] = f(a[i], b[i]);
  }

}

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.elemwise_add")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  DLTensor *args0 = args[0];
  DLTensor *args1 = args[1];
  DLTensor *args2 = args[2];
  
  auto f = [](int32_t a, int32_t b) -> int32_t {
    return a + b; 
  };
  elemwise(args0, args1, args2, f);
  print_to_file(args2, "elemwise_add.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.elemwise_sub")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  DLTensor *args0 = args[0];
  DLTensor *args1 = args[1];
  DLTensor *args2 = args[2];

  auto f = [](int32_t a, int32_t b) -> int32_t {
    return a - b; 
  };
  elemwise(args0, args1, args2, f);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.clip")
.set_body([](CVMArgs args, CVMRetValue* rv){
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   void *_attr = args[2];
   auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
   auto& param = cvm::get<cvm::top::ClipParam>(attr->parsed);
   int32_t max = param.a_max;
   int32_t min = param.a_min;
   int32_t *x_data = static_cast<int32_t*>(x->data);
   int32_t *y_data = static_cast<int32_t*>(y->data);
#pragma omp parallel for
   for (uint64_t i = 0; i < getSize(x); i++) {
    y_data[i] = std::max(std::min(max, x_data[i]), min);
   }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.flatten")
    .set_body([](CVMArgs args, CVMRetValue* rv)
{
     DLTensor *x = args[0];
     DLTensor *y = args[1];
     int32_t* x_data = static_cast<int32_t*>(x->data);
     int32_t* y_data = static_cast<int32_t*>(y->data);
     if(x_data != y_data){
        memcpy(y_data, x_data, getSize(x)*sizeof(int32_t));
     }

  print_to_file(y, "flatten.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.reshape")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  if(x->data == y->data) return;
  std::memcpy(y->data, x->data, getSize(x) * sizeof(int32_t));
  print_to_file(y, "reshape.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.cvm_clip")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  int32_t *x_data = static_cast<int32_t*>(x->data);
  int32_t *y_data = static_cast<int32_t*>(y->data);

  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::CVMClipParam>(attr->parsed);
  int32_t precision = param.precision;
  int32_t min = -(((int64_t)1 << (precision-1))-1);
  int32_t max = -min;
  uint64_t size = getSize(x);

  for(uint64_t i = 0; i < size; ++i){
    int32_t tmp = x_data[i];
    if (tmp > max) tmp = max;
    else if (tmp < min) tmp = min;
    y_data[i] = tmp;
  }
  print_to_file(y, "clip.txt");
}
);

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.cvm_right_shift")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *a = args[0];
    DLTensor *c = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::CVMRightShiftParam>(attr->parsed);
    int32_t precision = param.precision;
    int32_t b = param.shift_bit;
    int32_t* a_data = static_cast<int32_t*>(a->data);
    int32_t* c_data = static_cast<int32_t*>(c->data);
    int32_t min = -(((int64_t)1 << (precision-1)) - 1);
    int32_t max = -min;
    auto size = getSize(a);

    if (max == 0) {
      memset(c_data, 0, size * sizeof(int32_t));
    } else if (b >= 31) {
      for (uint64_t i = 0; i < size; ++i) c_data[i] = a_data[i] >> 31;
    } else if (precision + b >= 32) {
      for (uint64_t i = 0; i < size; ++i) {
        int t = a_data[i] >> (b - 1);
        c_data[i] = (t + 1) >> 1;
      }
    } else {
      int x_min = (-2 * max - 1) << (b - 1);
      int x_max = (2 * max - 1) << (b - 1);
      for (size_t i = 0; i < size; ++i) {
        int32_t t = a_data[i];
        if (t < x_min) c_data[i] = min;
        else if (t > x_max) c_data[i] = max;
        else c_data[i] = ((t >> (b - 1)) + 1) >> 1;
      }
    }

    // if (b == 1) {
    //   for(uint64_t i = 0; i < size; i++){
    //     int32_t shift_a = (a_data[i] + 1) >> 1;
    //     if (shift_a > max) shift_a = max;
    //     else if (shift_a < min) shift_a = min;
    //     c_data[i] = shift_a;
    //   }
    // } else {
    //   b -= 1;
    //   {
    //     for(uint64_t i = 0; i < size; i++){
    //       c_data[i] = a_data[i] >> b;
    //       ++c_data[i];
    //       c_data[i] >>= 1;
    //     }
    //     for(uint64_t i = 0; i < size; i++){
    //       auto& shift_a = c_data[i];
    //       if (shift_a > max) shift_a = max;
    //       else if (shift_a < min) shift_a = min;
    //     }
    //   }
    // }

  print_to_file(c, "cvm_right_shift.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.cvm_left_shift")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *a = args[0];
    DLTensor *c = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::CVMLeftShiftParam>(attr->parsed);
    int32_t precision = param.precision;
    int32_t b = param.shift_bit;std::string str_precision = args[2];
    int32_t* a_data = static_cast<int32_t*>(a->data);
    int32_t* c_data = static_cast<int32_t*>(c->data);
    int32_t min = -(((int64_t)1 << (precision-1)) - 1);
    int32_t max = -min;

    for(uint64_t i = 0; i < getSize(a); i++){
      int32_t shift_a = a_data[i] << b;
      c_data[i] = std::max(std::min(shift_a, max), min);
    }
});
}
}
