#ifndef CVM_OPS_H
#define CVM_OPS_H

#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/serializer.h>

#include <cvm/op.h>
#include <cvm/top/tensor.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "../../graph_runtime.h"

inline uint64_t getSize(DLTensor *dlTensor){
  uint64_t size = 1;
  for(int i = 0; i < dlTensor->ndim; i++){
      size *= dlTensor->shape[i];
  }
  return size;
}

// #define CVM_PROFILING
// #define CVM_PRINT_OP_RESULT

const std::string DIR = "/tmp/zkh/random_3_1/";
inline void print_to_file(DLTensor *y, std::string filename){
#if defined(CVM_PRINT_OP_RESULT)
  FILE *fp = fopen((DIR + filename).c_str(), "a+");
  int32_t *y_data = static_cast<int32_t*>(y->data);

  int32_t min = y_data[0], max= y_data[0];
  for(uint64_t i = 0; i < getSize(y); i++){
      min = min > y_data[i] ? y_data[i] : min;
      max = max < y_data[i] ? y_data[i] : max;
  }
  fprintf(fp, "%d %d\n", min, max);
  for(uint64_t i = 0; i < 20 && i < getSize(y); i++){
      fprintf(fp, "%d ", y_data[i]);
  }
  fprintf(fp, "\n");
  fclose(fp);
#endif
}

#endif
