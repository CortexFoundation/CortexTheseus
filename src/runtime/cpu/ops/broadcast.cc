#include "ops.h"

namespace cvm {
namespace runtime {

inline int32_t broadcast_i_index(int64_t* oshape, uint64_t o_index, int64_t* ishape, int idim, int odim){
  if(idim == 1 && ishape[0] == 1) return 0;
  uint64_t index = 0;
  uint64_t allIndex = 1;
  for(int i = 0; i < idim; i++){
    int idx = idim - 1 - i;
    int ovar = o_index % oshape[idx+odim-idim];
    if(ovar < ishape[idx]){
      index += allIndex * ovar;
    }
    allIndex =  allIndex * ishape[idx];
    o_index /= oshape[idx + odim-idim];
  }
  return index;
}

typedef std::function<int32_t(int32_t a, int32_t b)> broadcast_func;

void broadcast(DLTensor *args0, DLTensor* args1, DLTensor* args2, broadcast_func const &f){
  int32_t *a = static_cast<int32_t*>(args0->data);
  int32_t *b = static_cast<int32_t*>(args1->data);
  int32_t *c = static_cast<int32_t*>(args2->data);

  uint64_t a_size = getSize(args0);
  uint64_t b_size = getSize(args1);
  uint64_t c_size = getSize(args2);

  if(a_size == 1){
    for(uint64_t i = 0; i < c_size; ++i) c[i] = f(a[0], b[i]);
  } else if (b_size == 1) {
    for(uint64_t i = 0; i < c_size; ++i) c[i] = f(a[i], b[0]);
  }else{
#pragma omp parallel for
    for(uint64_t i = 0; i < c_size; ++i){
      int64_t a_index = broadcast_i_index(
          args2->shape, i, args0->shape, args0->ndim, args2->ndim);
      int64_t b_index = broadcast_i_index(
          args2->shape, i, args1->shape, args1->ndim, args2->ndim);
      c[i] = f(a[a_index], b[b_index]);
    }
  }
}

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.broadcast_add")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a + b;
    };

    broadcast(args0, args1, args2, f);
    print_to_file(args2, "broadcast_add.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.broadcast_sub")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a - b;
    };

    broadcast(args0, args1, args2, f);
    print_to_file(args2, "broadcast_sub.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.broadcast_mul")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a * b;
    };

    broadcast(args0, args1, args2, f);
    print_to_file(args2, "broadcast_mul.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.broadcast_max")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a > b ? a : b;
    };

    broadcast(args0, args1, args2, f);
    print_to_file(args2, "broadcast_max.txt");
});
CVM_REGISTER_GLOBAL("cvm.runtime.cpu.broadcast_div")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return b == 0 ? 0 : a/b;
    };

    broadcast(args0, args1, args2, f);
    print_to_file(args2, "broadcast_div.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cpu.broadcast_greater")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *args0 = args[0];
    DLTensor *args1 = args[1];
    DLTensor *args2 = args[2];
    broadcast_func f = [](int32_t a, int32_t b) -> int32_t {
      return a > b;
    };

    broadcast(args0, args1, args2, f);
    print_to_file(args2, "broadcast_greater.txt");
});
}
}

