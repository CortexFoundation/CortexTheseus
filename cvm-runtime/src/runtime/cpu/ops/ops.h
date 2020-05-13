#ifndef CVM_OPS_H
#define CVM_OPS_H

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
#include <omp.h>

inline uint64_t getSize(DLTensor *dlTensor){
  uint64_t size = 1;
  for(int i = 0; i < dlTensor->ndim; i++){
      size *= dlTensor->shape[i];
  }
  return size;
}

namespace cvm{
namespace runtime {
// #define CVM_PRINT_OP_RESULT

const std::string DIR = "/tmp/zkh/ssd/cpu/";
inline void print_to_file(DLTensor *y, std::string filename, bool all=false){
#if defined(CVM_PRINT_OP_RESULT)
  FILE *fp = fopen((DIR + filename).c_str(), "a+");
  int32_t *y_data = static_cast<int32_t*>(y->data);

  int32_t min = y_data[0], max= y_data[0];
  for(uint64_t i = 0; i < getSize(y); i++){
      min = min > y_data[i] ? y_data[i] : min;
      max = max < y_data[i] ? y_data[i] : max;
  }
  fprintf(fp, "%d %d\n", min, max);
  if(all){
    for(uint64_t i = 0; i < getSize(y); i++){
      fprintf(fp, "%d ", y_data[i]);
    }
  }else{
    for(uint64_t i = 0; i < 5000 && i < getSize(y); i++){
      fprintf(fp, "%d ", y_data[i]);
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
#endif
}

}
}

#endif
