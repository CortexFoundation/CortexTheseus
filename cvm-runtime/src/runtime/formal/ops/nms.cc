#include <iostream>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <cvm/top/nn.h>

#include "ops.h"

namespace cvm {
namespace runtime{

int64_t iou(const int32_t *rect1, const int32_t *rect2, const int32_t format){
    int32_t x1_min = format == FORMAT_CORNER ? rect1[0] : rect1[0] - rect1[2]/2;
    int32_t y1_min = format == FORMAT_CORNER ? rect1[1] : rect1[1] - rect1[3]/2;
    int32_t x1_max = format == FORMAT_CORNER ? rect1[2] : x1_min + rect1[2];
    int32_t y1_max = format == FORMAT_CORNER ? rect1[3] : y1_min + rect1[3];

    int32_t x2_min = format == FORMAT_CORNER ? rect2[0] : rect2[0] - rect2[2]/2;
    int32_t y2_min = format == FORMAT_CORNER ? rect2[1] : rect2[1] - rect2[3]/2;
    int32_t x2_max = format == FORMAT_CORNER ? rect2[2] : x2_min + rect2[2];
    int32_t y2_max = format == FORMAT_CORNER ? rect2[3] : y2_min + rect2[3];

    //x1,x2,y1,y2 precision <= 30
    //sum_arrea precision<=63
    int64_t sum_area = static_cast<int64_t>(x1_max-x1_min) * (y1_max-y1_min) + 
        static_cast<int64_t>(x2_max-x2_min) * (y2_max-y2_min);
    if (sum_area <= 0) return 0;

    //w,h precision <= 31
    int32_t w = std::max(0, std::min(x1_max, x2_max) - std::max(x1_min, x2_min));
    int32_t h = std::max(0, std::min(y1_max, y2_max) - std::max(y1_min, y2_min));
    //overlap_area precision <= 62
    int64_t overlap_area = static_cast<int64_t>(h)*w;
    //tmp precision <= 63
    int64_t tmp = (sum_area - overlap_area);
    if (tmp <= 0) return 0;

    int64_t max64 = ((uint64_t)1 << 63) - 1;
    if (max64 / 100 < overlap_area) { tmp /= 100; } 
    else { overlap_area *= 100; }

    return overlap_area / tmp;
}

void get_valid_count(const int32_t *x_data, int32_t *y_data, int32_t *valid_count_data, const int32_t batchs, const int32_t n, const int32_t k, const int32_t score_threshold){
  for(int32_t i = 0; i < batchs; i++){
      int32_t y_index = 0;
      const int32_t *input = x_data + i * n * k;
      int32_t *output = y_data + i * n * k;
      for(int32_t j = 0; j < n; j++){
          const int32_t *row = input + j * k;
          if(row[1] > score_threshold){
              memcpy(&output[y_index * k], row, k * sizeof(int32_t));
              y_index += 1;
          }
      }
      valid_count_data[i] = y_index;
      if(y_index < n){
          memset(&output[y_index * k], -1, (n-y_index) * k * sizeof(int32_t));
      }
  }
}

CVM_REGISTER_GLOBAL("cvm.runtime.formal.get_valid_counts")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *x = args[0];
    DLTensor *valid_count = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::GetValidCountsParam>(attr->parsed);

    int32_t score_threshold = param.score_threshold; 

    int32_t batches = x->shape[0];
    int32_t n = x->shape[1];
    int32_t k = x->shape[2];

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *valid_count_data = static_cast<int32_t*>(valid_count->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    get_valid_count(x_data, y_data, valid_count_data, batches, n, k, score_threshold);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.non_max_suppression")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    auto X = CVMArg2Data<int32_t>(args[0]);
    auto valid_count = CVMArg2Data<int32_t>(args[1]);
    auto Y = CVMArg2Data<int32_t>(args[2]);
    auto params = CVMArg2Attr<top::NonMaximumSuppressionParam>(args[3]);

    // X's shape must be (B, N, K), K = 6
    auto x_shape = CVMArgShape(args[0]);
    int32_t B = x_shape[0];
    int32_t N = x_shape[1];
    int32_t K = x_shape[2];

    for (int32_t b = 0; b < B; ++b) {
      int32_t T = std::max(std::min(N, valid_count[b]), 0);
      std::vector<int32_t*> R(T); // sorted X in score descending order
      for (int i = 0; i < T; ++i) R[i] = X + b * N * K + i * K;

      std::stable_sort(R.begin(), R.end(), 
        [](const int32_t* a, const int32_t* b) -> bool {
            return a[1] > b[1];
        });

      int32_t n_max = T; // n_max = min{T, MOS}
      if (params.max_output_size >= 0)
        n_max = std::min(n_max, params.max_output_size);
      int32_t p_max = T; // p_max = min{TK, T}
      if (params.top_k >= 0) 
        p_max = std::min(p_max, params.top_k);

      int32_t n = 0; // dynamic calculate union U, as Y index.
      int32_t *y_batch = Y + b * N * K; // temporary variable
      // dynamic calculate U, and n \in [0, min{n_max, card{U})
      for (int32_t p = 0; n < n_max && p < p_max; ++p) { // p \in [0, p_max)
        if (R[p][0] < 0) continue; // R[b, p, 0] >= 0

        bool ignored = false; // iou(p, q) <= iou_threshold, \forall q in U.
        for (int32_t i = 0; i < n; ++i) {
          if (params.force_suppress || y_batch[i*K+0] == R[p][0]) {
            int64_t iou_ret = iou(y_batch+i*K+2, R[p]+2, FORMAT_CORNER);
            if (iou_ret >= params.iou_threshold) {
              ignored = true;
              break;
            }
          }
        }

        if (!ignored) { // append U: copy corresponding element to Y.
          memcpy(y_batch+n*K, R[p], K*sizeof(int32_t));
          ++n;
        }
      }

      memset(y_batch+n*K, -1, (N-n)*K*sizeof(int32_t)); // others set -1.
    }
});

}
}
