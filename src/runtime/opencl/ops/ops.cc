#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>

#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "ops.h"

namespace cvm{
  namespace runtime{

inline uint64_t getSize(DLTensor *dlTensor){
  uint64_t size = 1;
  for(int i = 0; i < dlTensor->ndim; i++){
    size *= dlTensor->shape[i];
  }
  return size;
}

CVM_REGISTER_GLOBAL("cvm.runtime.opencl.elemwise_add")
  .set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
      DLTensor *a = args[0];
      DLTensor *b = args[1];
      DLTensor *c = args[2];
      void *a_data = (a->data);
      void *b_data = (b->data);
      void *c_data = (c->data);
      uint64_t n = getSize(a);
      int error_code = 0;
      opencl_elemwise_add(a_data, b_data, c_data, n);

});


}
}
