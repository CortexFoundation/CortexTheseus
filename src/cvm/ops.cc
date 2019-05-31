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

#include "omp.h"
#include <immintrin.h>

#include "graph_runtime.h"

namespace cvm {
namespace runtime {

double transpose_int8_avx256_transpose_cnt = 0;
double transpose_int8_avx256_gemm_cnt = 0;
double im2col_cnt = 0;
double cvm_op_rightshift_cnt = 0;
double cvm_op_clip_cnt = 0;
double cvm_op_dense_cnt = 0;
double cvm_op_maxpool_cnt = 0;
double cvm_op_broadcast_cnt = 0;
double cvm_op_concat_cnt = 0;
double cvm_op_upsampling_cnt = 0;

#define CVM_PROFILING

inline uint64_t getSize(DLTensor *dlTensor){
    uint64_t size = 1;
    for(int i = 0; i < dlTensor->ndim; i++){
        size *= dlTensor->shape[i];
    }
    return size;
}

/**
* x
* y
* a_min -127
* a_max 127
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.clip").set_body([](CVMArgs args, CVMRetValue* rv) {
   VERIFY(args.num_args == 3);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   void *_attr = args[2];
   auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
   auto& param = cvm::get<cvm::top::ClipParam>(attr->parsed);
   int max = param.a_max;
   int min = param.a_min;
   for (uint64_t i = 0; i < getSize(x); i++) {
 		static_cast<int32_t*>(y->data)[i] = std::max(std::min(max, static_cast<int32_t*>(x->data)[i]), min);
   }
 });

 CVM_REGISTER_GLOBAL("cvm.runtime.cvm.relu").set_body([](CVMArgs args, CVMRetValue* rv) {
   VERIFY(args.num_args == 3);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
#pragma omp parallel for
   for (uint64_t i = 0; i < getSize(x); i++) {
        auto tmp = static_cast<int32_t*>(x->data)[i];
        if (tmp < 0)
            tmp = 0;
 		static_cast<int32_t*>(y->data)[i] = tmp;
   }
 });

/*
* x
* w
* b
* y
* units 1000
* use_bias True
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.dense").set_body([](CVMArgs args, CVMRetValue* rv) {
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
  int ndim = args.num_args;
  VERIFY(ndim == 5 || ndim == 4);
  DLTensor *x = args[0];
  DLTensor *w = args[1];
  DLTensor *b = nullptr;
  DLTensor *y = nullptr;
  int32_t* db = nullptr;
  if(ndim == 5){
	b = args[2];
    VERIFY(b->ndim == 1) << "dense requires 1-D bias";
	y = args[3];
    db = static_cast<int32_t*>(b->data);
  } else{
    y = args[2];
  }
  VERIFY(x->ndim == 2) << "dense requires 2-D data";
  VERIFY(w->ndim == 2) << "dense reuqires 2-D weight";

  auto dx = static_cast<int32_t*>(x->data);
  auto dy = static_cast<int32_t*>(y->data);
  auto dw = static_cast<int32_t*>(w->data);
  auto N = y->shape[1], K = x->shape[1];
  int blocks = K / 32 * 32;
  // std::cerr << y->shape[0] << " " << y->shape[1] << "\n";
  // std::cerr << x->shape[0] << " " << x->shape[1] << "\n";
  // std::cerr << w->shape[0] << " " << w->shape[1] << "\n";
  int32_t weight_size = w->shape[0] * w->shape[1];
  std::unique_ptr<int8_t> int8_filter(new int8_t[sizeof(int8_t) * weight_size]);
  if(!int8_filter) {
      CHECK(false) << "create buffer int8_filter failed";
  }

  for(int32_t i = 0; i < weight_size; i++){
      *(int8_filter.get() + i) = static_cast<int8_t>(dw[i]);
  }

  int32_t x_size = x->shape[0] * x->shape[1];
  std::unique_ptr<int8_t> int8_x(new int8_t[sizeof(int8_t) * x_size]);
  if(!int8_x) {
      CHECK(false) << "create buffer int8_x failed";
  }
  bool all_positive = true;
  for(int32_t i = 0; i < x_size; i++){
      int8_x.get()[i] = static_cast<int8_t>(dx[i]);
      if ((int8_x.get()[i]) < 0)
          all_positive = false;
  }
  // std::cerr << "all_positive = " << all_positive << "\n";

  int16_t int16[16];
  for(int i = 0; i < 16; i++)
      int16[i] = 1;
  __m256i vint16 = _mm256_loadu_si256((__m256i*)&int16);

  for (uint32_t di = 0; di < y->shape[0]; di++) {
      auto cdy = dy + di * N;
      auto ap_outer = int8_x.get() + di * K;
#pragma omp parallel for
      for (uint32_t oi = 0; oi < N; oi++) {
          auto bp_inner = int8_filter.get() + oi * K;
          auto ap_inner = ap_outer;
          int sum = 0;

          int k = 0;
          if (all_positive) {
              __m256i vc = _mm256_setzero_si256();
              for(k = 0; k < blocks; k+=32, ap_inner+=32, bp_inner+=32){
                  __m256i va = _mm256_loadu_si256((__m256i*)bp_inner);
                  __m256i vb = _mm256_loadu_si256((__m256i*)ap_inner);
                  __m256i vresult1 = _mm256_maddubs_epi16(vb, va);
                  __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
                  vc = _mm256_add_epi32(vresult2, vc);
              }
              for(int ti = 0; ti < 8; ti++){
                  sum += ((int32_t*)&vc)[ti];
              }
          }

          // remained part
          for(; k < K; k++){
              sum += ap_inner[k] * bp_inner[k];
          }
          if(db != nullptr){
              sum += db[oi];
          }
          cdy[oi] = sum;
      }
  }

#ifdef CVM_PROFILING
        cvm_op_dense_cnt += omp_get_wtime() - start;
#endif
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.flatten").set_body([]
(CVMArgs args, CVMRetValue* rv){
     VERIFY(args.num_args == 3);
     DLTensor *x = args[0];
     DLTensor *y = args[1];
     for (uint64_t i = 0; i < getSize(x); i++) {
         static_cast<int32_t*>(y->data)[i] = static_cast<int32_t*>(x->data)[i];
     }
});

bool transpose_int8_avx256(const int8_t *a, const int8_t *b, const int32_t *bias,
        int32_t *c, const int M, const int K, const int N){
#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
    int8_t *tr_b = (int8_t*)malloc(sizeof(int8_t) * K*N);
    if (tr_b == NULL) return false;

    int i = 0, j = 0;
    const int32_t tK = K / 32 * 32;
    const int32_t tN = N / 32 * 32;
    for(i = 0; i < tK; i+=32){
        for(j = 0; j < tN; j+=32){
            int8_t tile[32][32];
            for(int ti = 0; ti < 32; ti++){
                for(int tj = 0; tj < 32; tj++){
                    tile[tj][ti] = b[(i+ti)*N + j+tj];
                }
            }
            for(int ti = 0; ti < 32; ti++){
                for(int tj = 0; tj < 32; tj++){
                    tr_b[(j+ti) * K + i + tj] = tile[ti][tj];
                }
            }
        }
        for(int ti = 0; ti < 32; ti++){
            for(int tj = j; tj < N; tj++){
                tr_b[tj * K + i+ti] = b[(i+ti) * N + tj];
            }
        }
    }
    for(; i < K; i++){
        for(j = 0; j < N; j++){
            tr_b[j * K + i] = b[i * N + j];
        }
    }
#ifdef CVM_PROFILING
    transpose_int8_avx256_transpose_cnt += omp_get_wtime() - start;
    start = omp_get_wtime();
#endif
    int16_t int16[16];
    for(int i = 0; i < 16; i++) int16[i] = 1;
    __m256i vint16 = _mm256_loadu_si256((__m256i*)&int16);
    int8_t ap [32], bp[32];
    memset(ap, 0, sizeof(ap));
    memset(bp, 0, sizeof(bp));

    int blocks = K / 32 * 32;
if (K % 32 == 0) {
#pragma omp parallel for
    for(int i = 0; i < M; i++){
        int32_t bV = bias != NULL ? bias[i] : 0;
        for(int j = 0; j < N; j++){
            __m256i vc = _mm256_setzero_si256();
            int k = 0;
            auto ap_inner = a + i * K;
            auto bp_inner = tr_b + j * K;
            for(k = 0; k < blocks; k+=32, ap_inner+=32, bp_inner+=32){
                __m256i va = _mm256_loadu_si256((__m256i*)(ap_inner));
                __m256i vb = _mm256_loadu_si256((__m256i*)bp_inner);
                __m256i vresult1 = _mm256_maddubs_epi16(vb, va);
                __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
                vc = _mm256_add_epi32(vresult2, vc);
            }
            int sum = 0;
            for(int ti = 0; ti < 8; ti++){
                sum += ((int32_t*)&vc)[ti];
            }
            c[i*N+j] = sum + bV;
        }
    }
} else {
    for(int i = 0; i < M; i++){
        int32_t bV = bias != NULL ? bias[i] : 0;
        for(int j = 0; j < N; j++){
            __m256i vc = _mm256_setzero_si256();
            int k = 0;
            auto ap_inner = a + i * K;
            auto bp_inner = tr_b + j * K;
            for(k = 0; k < blocks; k+=32, ap_inner+=32, bp_inner+=32){
                __m256i va = _mm256_loadu_si256((__m256i*)(ap_inner));
                __m256i vb = _mm256_loadu_si256((__m256i*)bp_inner);
                __m256i vresult1 = _mm256_maddubs_epi16(vb, va);
                __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
                vc = _mm256_add_epi32(vresult2, vc);

            }
            if (K % 32 != 0) {
                memcpy(ap, ap_inner, sizeof(int8_t) * (K - k));
                memcpy(bp, bp_inner, sizeof(int8_t) * (K - k));
                {
                    __m256i va = _mm256_loadu_si256((__m256i*)ap);
                    __m256i vb = _mm256_loadu_si256((__m256i*)bp);
                    __m256i vresult1 = _mm256_maddubs_epi16(vb, va);
                    __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
                    vc = _mm256_add_epi32(vresult2, vc);
                }
                k = K;
            }
            int sum = 0;
            for(int ti = 0; ti < 8; ti++){
                sum += ((int32_t*)&vc)[ti];
            }
            c[i*N+j] = sum + bV;
        }
    }

}

    free(tr_b);
#ifdef CVM_PROFILING
    double et = omp_get_wtime() - start;
    // std::cerr << "gemm " << N << " " << M << " " << K << " " << et * 1000 << " " << N * M * K / 1024.0 / 1024.0 << "\n";
    transpose_int8_avx256_gemm_cnt += et;
#endif
    return true;
}
void matrix_mul(const int8_t *a, const int8_t *b, const int32_t *bias,
        int32_t *c, const int M, const int K, const int N){
    std::memset(c, 0, sizeof(int32_t) * M * N);
#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
#pragma omp parallel for
    for(int i = 0; i < M; i++){
        for(int k = 0; k < K; k++){
           int32_t aV = static_cast<int32_t>(a[i * K + k]);
            for(int j = 0; j < N; j++){
                c[i*N + j] += aV * static_cast<int32_t>(b[k*N + j]);
            }
        }
    }
    if(bias != NULL){
        for(int i = 0; i < M; i++){
            register int32_t biasV = bias[i];
            for(int j = 0; j < N; j++){
                c[i*N+j] += biasV;
            }
        }
    }
#ifdef CVM_PROFILING
        cvm_op_dense_cnt += omp_get_wtime() - start;
#endif
}
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
void im2col_cpu(const int32_t* data_im, const int channels,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        int8_t* data_col, bool &flag) {
#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
    auto data_col_init = data_col;
    const int output_h = (height + 2 * pad_h -
            (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
            (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                int32_t tv = data_im[input_row * width + input_col];
                                if(tv < 0) flag = true;
                                *(data_col++) = static_cast<int8_t>(tv);
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
        // std::cout << "inchannel = " << channel
        //           << " " << data_col -  data_col_init << " "
        //           << "hw = " << height << ", " << width << " " << stride_h << " " <<  stride_w
        //           << "\n";
    }
#ifdef CVM_PROFILING
    im2col_cnt +=  omp_get_wtime() - start;
#endif
}

inline void depthwise_conv2d(
        int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
        int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
        int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
        int32_t *b_data,
        int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
        int32_t groups){
    for(int n = 0; n < n_batch; ++n){
        for(int c = 0; c < in_channels; ++c){
            for(int h = 0; h < o_h; ++h){
                for(int w = 0; w < o_w; ++w){
                    int32_t sum = 0;
                    for(int fh = 0; fh < filter_h; ++fh){
                        for(int fw = 0; fw < filter_w; ++fw){
                            int th = h * stride_h + fh*dilation_h - padding[0];
                            int tw = w * stride_w + fw*dilation_w - padding[1];
                            if(th < 0 || tw < 0 || th >= x_h || tw >= x_w)
                                continue;
                            sum += x_data[n * in_channels * x_h * x_w + c * x_h * x_w + th * x_w + tw]
                                * w_data[c * filter_h * filter_w + fh * filter_w + fw];
                        }
                    }
                    y_data[n * in_channels * o_h * o_w + c * o_h * o_w + h * o_w + w] = sum + (b_data != nullptr ? b_data[c] : 0);
                }
            }
        }
    }
}
/*
input
weight
bias
output
groups 1
dilation (1, 1)
channels 512
layout NCHW
kernel_layout OIHW
kernel_size [1, 1]
padding (0, 0)
use_bias True
strides (1, 1)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.conv2d").set_body([]
 (CVMArgs args, CVMRetValue* rv){
    VERIFY(args.num_args == 4 || args.num_args == 5);
    DLTensor *x = args[0];
    VERIFY(x->ndim == 4);
    DLTensor *w = args[1];
    VERIFY(w->ndim == 4);
	DLTensor *b = nullptr; //args[2];
    DLTensor *y = nullptr;
    void *_attr;

    if(args.num_args == 5){
      b = args[2];
      y = args[3];
      _attr = args[4];
    } else {
      y = args[2];
      _attr = args[3];
    }
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::Conv2DParam>(attr->parsed);
    int groups = param.groups;
	int dilation[2] = {(int)param.dilation[0], (int)param.dilation[1]};
    //TODO(@kaihuo) check kernel_size == w->shape
	// int kernel_size[2] = {(int)param.kernel_size[0], (int)param.kernel_size[1]};
	int padding[2] = {(int)param.padding[0], (int)param.padding[1]};
	int strides[2] = {(int)param.strides[0], (int)param.strides[1]};

    int stride_h = strides[0];
    int stride_w = strides[1];
    int dilation_h = dilation[0];
    int dilation_w = dilation[1];

    int32_t* x_data = (int32_t*)x->data;
    int32_t* w_data = (int32_t*)w->data;
    int32_t* y_data = (int32_t*)y->data;
	int32_t* b_data = b != nullptr ? (int32_t*)b->data : nullptr;

    int out_channels = static_cast<int>(w->shape[0]);
    int filter_c = static_cast<int>(w->shape[1]);
    int filter_h = static_cast<int>(w->shape[2]);
    int filter_w = static_cast<int>(w->shape[3]);
    filter_h = (filter_h - 1) * dilation[0] + 1;
    filter_w = (filter_w - 1) * dilation[1] + 1;

    int n_batch = static_cast<int>(x->shape[0]);
    int in_channels = static_cast<int>(x->shape[1]);
    int x_h = static_cast<int>(x->shape[2]);
    int x_w = static_cast<int>(x->shape[3]);
	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
    if(n_batch < 1 || in_channels < 1 || x_h < 1 || x_w < 1 || filter_c < 1 || filter_h < 1 || filter_w < 1 ||
            padding[0] < 0 || padding[1] < 0 || stride_h < 1 || stride_w < 1 || dilation_h < 1 || dilation_w < 1 ||
             out_channels < 1 || o_h < 1 || o_w < 1){
        VERIFY(false) << "error args";
    }

    if(groups > 1){
        depthwise_conv2d(
                x_data, n_batch, in_channels, x_h, x_w,
                w_data, filter_c, filter_h, filter_w,
                y_data, out_channels, o_h, o_w,
                b_data,
                padding, stride_h, stride_w, dilation[0], dilation[1],
                groups);
    }else{
        int8_t *data_col = (int8_t*)malloc(sizeof(int8_t) * in_channels * filter_h * filter_w * o_h * o_w);
        if(data_col == NULL){
            CHECK(false) << "malloc failed.";
        }
        int32_t fn = out_channels * in_channels * filter_h * filter_w;
        int8_t *int8_filter = (int8_t*)malloc(sizeof(int8_t) * fn);
        if(int8_filter == NULL){
            free(data_col);
            CHECK(false);
        }

        for(int32_t i = 0; i < fn; i++){
            int8_filter[i] = static_cast<int8_t>(w_data[i]);
        }

        for(int i = 0; i < n_batch; i++){
            bool flag = false;
            im2col_cpu(x_data + i * in_channels * x_h * x_w, in_channels, x_h, x_w, filter_h, filter_w, padding[0], padding[1],
                    stride_h, stride_w, dilation_h, dilation_w, data_col, flag);
            const int M = out_channels;
            const int K = in_channels * filter_h * filter_w;
            const int N = o_h * o_w;
            if(flag){
                matrix_mul(int8_filter, data_col, b_data, y_data + i * out_channels * o_h * o_w,
                    M, K, N);
            }else{
                transpose_int8_avx256(int8_filter, data_col, b_data, y_data + i * out_channels * o_h * o_w,
                    M, K, N);
            }
        }
        free(data_col);
        free(int8_filter);
    }

 });

inline int32_t broadcast_i_index(int64_t* oshape, uint64_t o_index, int64_t* ishape, int idim){
    if(idim == 1 && ishape[0] == 1) return 0;
    uint64_t index = 0;
    uint64_t allIndex = 0;
    for(int i = 0; i < idim; i++){
        int idx = idim - 1 - i;
        int ovar = o_index % oshape[idx];
        if(ovar < ishape[idx]){
            index += i == 0 ? ovar : allIndex * ovar;
        }
        allIndex = (i == 0 ? ishape[idim-1] : allIndex * ishape[idx]);
        o_index /= oshape[idx];
    }
    return index;
}

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_add")
    .set_body([](CVMArgs args, CVMRetValue *ret){
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        if(args1->ndim == 1){
            for(uint64_t i = 0; i < getSize(args0); ++i){
                c[i] = a[i] + b[0];
            }
        }else{
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args0); ++i){
                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
                int64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
                int64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
                c[i] = a[a_index] + b[b_index];
            }
        }
#ifdef CVM_PROFILING
        cvm_op_broadcast_cnt += omp_get_wtime() - start;
#endif
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_sub")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        if(args1->ndim == 1){
            for(uint64_t i = 0; i < getSize(args0); ++i){
                c[i] = a[i] - b[0];
            }
        }else{
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args0); ++i){
                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
                c[i] = a[a_index] - b[b_index];
            }
        }
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_mul")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        if(args1->ndim == 1){
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args0); ++i){
                c[i] = a[i] * b[0];
            }
        }else{
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args0); ++i){
                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
                c[i] = a[a_index] * b[b_index];
            }
        }
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_div")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        if(args1->ndim == 1){
            VERIFY(b[0] != 0);
            for(uint64_t i = 0; i < getSize(args0); ++i){
                c[i] = a[i] / b[0];
            }
        }else{
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args0); ++i){
                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
                VERIFY(b[b_index] != 0);
                c[i] = a[a_index] / b[b_index];
            }
        }
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_right_shift")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

        if(args1->ndim == 1){
            for(uint64_t i = 0; i < getSize(args0); ++i){
                c[i] = a[i] >> b[0];
            }
        }else{
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args0); ++i){
                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
                c[i] = a[a_index] >> b[b_index];
            }
        }
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_left_shift")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        if(args1->ndim == 1){
            for(uint64_t i = 0; i < getSize(args0); ++i){
                c[i] = a[i] << b[0];
            }
        }else{
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(args0); ++i){
                uint64_t o_index = i;//broadcast_o_index(args2->shape, args2->ndim, o_index);
                uint64_t a_index = broadcast_i_index(args2->shape, o_index, args0->shape, args0->ndim);
                uint64_t b_index = broadcast_i_index(args2->shape, o_index, args1->shape, args1->ndim);
                c[i] = a[a_index] << b[b_index];
            }
        }
    });

/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.max_pool2d")
    .set_body([](CVMArgs args, CVMRetValue *ret){
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
    VERIFY(args.num_args == 3);
	DLTensor *x = args[0];
	DLTensor *y = args[1];
	void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::MaxPool2DParam>(attr->parsed);
	int strides[2] = {(int)param.strides[0], (int)param.strides[1]};
	int pool_size[2] = {(int)param.pool_size[0], (int)param.pool_size[1]};
	int padding[2] = {(int)param.padding[0], (int)param.padding[1]};

    int stride_h = strides[0];
    int stride_w = strides[1];

    int32_t* x_data = (int32_t*)x->data;
    int32_t* y_data = (int32_t*)y->data;

    int filter_h = pool_size[0];
    int filter_w = pool_size[1];

    int n_batch = static_cast<int>(x->shape[0]);
    int in_channels = static_cast<int>(x->shape[1]);
    int out_channels = in_channels;
    int x_h = static_cast<int>(x->shape[2]);
    int x_w = static_cast<int>(x->shape[3]);
//  int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
//  int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
    int o_h = static_cast<int>(y->shape[2]);
    int o_w = static_cast<int>(y->shape[3]);
#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
#define GETW(o, i, h, w) w_data[(o) * in_channels * filter_h * filter_w + (i) * filter_h * filter_w + (h) * filter_w + (w)]
#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
    auto calc_func = [&](int n, int k, int p, int q) {
    int y_sum = int(1)<<31;
    for (int r = 0; r < filter_h; ++r) {
      for (int s = 0; s < filter_w; ++s) {
        auto tp = p * stride_h + r - padding[0];
        auto tq = q * stride_w + s - padding[1];
        int32_t x_tmp = 0;
        if (!(tp < 0 || tq < 0 || tp >= x_h || tq >= x_w))
          x_tmp = GETX(n, k, tp, tq);
        y_sum = std::max(x_tmp, y_sum);
      }
    }
    return y_sum;

  };
    for (int n = 0; n < n_batch; ++n) {
        for (int k = 0; k < out_channels; ++k) {
            for (int p = 0; p < o_h; ++p) {
                for (int q = 0; q < o_w; ++q) {
                    GETY(n, k, p, q) = calc_func(n, k, p, q);
                }
            }
        }
    }
#ifdef CVM_PROFILING
        cvm_op_maxpool_cnt += omp_get_wtime() - start;
#endif

});

/*
* axis (2, 3)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.sum")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
		DLTensor *x = args[0];
		DLTensor *y = args[1];
        //TODO(@kaihuo) unused axis, check
        //void *_attr = args[2];
        // auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
        //auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
		// int axis[2] = {(int)param.axis[0], (int)param.axis[1]};
		int32_t *x_data = static_cast<int32_t*>(x->data);
		int32_t *y_data = static_cast<int32_t*>(y->data);
		int n_batch = static_cast<int>(x->shape[0]);
		int channels = static_cast<int>(x->shape[1]);
		int x_h = static_cast<int>(x->shape[2]);
		int x_w = static_cast<int>(x->shape[3]);
		for(int i = 0; i < n_batch; i++){
			for(int j = 0; j < channels; j++){
				int32_t sum = 0;
				for(int h = 0; h < x_h; h++){
					for(int w = 0; w < x_w; w++){
						sum += x_data[i * channels * x_h * x_w + j * x_h * x_w + h * x_w + w];
					}
				}
				y_data[i*channels + j] = sum;
			}
		}
    });


CVM_REGISTER_GLOBAL("cvm.runtime.cvm.elemwise_add")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 4);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);

#pragma omp parallel for
        for(uint64_t i = 0; i < getSize(args0); i++){
            c[i] = a[i] + b[i];
        }
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.reshape")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
         // TODO(kaihuo) CHECK
		 // void *_attr = args[2];
         // auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
         // auto &param = cvm::get<cvm::top::ReshapeParam>(attr->parsed);
		 if(x->data == y->data) return;
		 std::memcpy(y->data, x->data, getSize(x) * sizeof(int32_t));
    });

/*\brief:
 * x, input data
 * y, output data
 * precision, clip precision
 */
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_clip")
    .set_body([](CVMArgs args, CVMRetValue *ret){
         VERIFY(args.num_args == 3);

#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
         DLTensor *x = args[0];
         DLTensor *y = args[1];
         int32_t *x_data = static_cast<int32_t*>(x->data);
         int32_t *y_data = static_cast<int32_t*>(y->data);

         void *_attr = args[2];
         auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
         auto &param = cvm::get<cvm::top::CVMClipParam>(attr->parsed);
	     int32_t precision = param.precision;
         VERIFY(precision > 0) << "precision must greater zero";
         int32_t min = -((1 << (precision-1))-1);
         int32_t max = -min;

#pragma omp parallel for
         for(uint64_t i = 0; i < getSize(x); i++){
             int& tmp = x_data[i];
             if (tmp > max)
                 tmp = max;
             if (tmp < min)
                 tmp = min;
             y_data[i] = tmp;
         }
#ifdef CVM_PROFILING
    cvm_op_clip_cnt += omp_get_wtime() - start;
#endif
    });

/*
 * a, input data
 * c, output data
 * precision, clip precision
 * b, shift b
 * */
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_right_shift")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *a = args[0];
        DLTensor *c = args[1];

#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
        void *_attr = args[2];
        auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
        auto &param = cvm::get<cvm::top::CVMRightShiftParam>(attr->parsed);
        int32_t precision = param.precision;
        int32_t b = param.shift_bit;
        int32_t* a_data = static_cast<int32_t*>(a->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        VERIFY_GT(precision, 0) << "precision must greater zero";
        int32_t min = -((1 << (precision-1)) - 1);
        int32_t max = -min;
        auto size = getSize(a);

        //TODO(kaihuo) check if b == 0 exists in symbol file
        if (b == 0) {
            memcpy(c_data, a_data, size * sizeof(int32_t));
        } else if (b == 1) {
            #pragma omp parallel for
            for(uint64_t i = 0; i < size; i++){
                int32_t shift_a = (a_data[i] + 1) >> 1;
                if (shift_a > max)
                    shift_a = max;
                if (shift_a < min)
                    shift_a = min;
                c_data[i] = shift_a;
            }
        } else {
            b -= 1;
            #pragma omp parallel
            {
            #pragma omp for
            for(uint64_t i = 0; i < size; i++){
                c_data[i] = a_data[i] >> b;
                ++c_data[i];
                c_data[i] >>= 1;
            }
            #pragma omp for
            for(uint64_t i = 0; i < size; i++){
                auto& shift_a = c_data[i];
                if (shift_a > max)
                    shift_a = max;
                if (shift_a < min)
                    shift_a = min;
                c_data[i] = shift_a;
            }
            }
        }

#ifdef CVM_PROFILING
    cvm_op_rightshift_cnt += omp_get_wtime() - start;
#endif
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_left_shift")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
        DLTensor *a = args[0];
        DLTensor *c = args[1];
        void *_attr = args[2];
        auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
        auto &param = cvm::get<cvm::top::CVMLeftShiftParam>(attr->parsed);
        int32_t precision = param.precision;
        int32_t b = param.shift_bit;std::string str_precision = args[2];
        int32_t* a_data = static_cast<int32_t*>(a->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        VERIFY_GT(precision, 0) << "precision must greater zero";
        int32_t min = -((1 << (precision-1)) - 1);
        int32_t max = -min;

        for(uint64_t i = 0; i < getSize(a); i++){
            int32_t shift_a = a_data[i];
            if(b == 0) c_data[i] = shift_a;
            else {
                shift_a = a_data[i] << b;
                c_data[i] = std::max(std::min(shift_a, max), min);
            }
        }
#ifdef CVM_PROFILING
    // cvm_op_requant_cnt += omp_get_wtime() - start;
#endif
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.log2")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
//        std::string x_str = args[0];
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t *x = static_cast<int32_t*>(dlx->data);
        VERIFY(x[0] != 0);
        for(int i = 0; i < 64; i++){
            int64_t tmp = (int64_t)1 << i;
            if(x[0] < tmp){
                y_data[0] = i;
                return;
            }
        }
        y_data[0] = 64;
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.__div_scalar__")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        void *_attr = args[2];
        auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
        auto &param = cvm::get<cvm::top::ScalarParam>(attr->parsed);
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t scalar = param.scalar;
        VERIFY(scalar != 0);
        int32_t* x = static_cast<int32_t*>(dlx->data);
        for(uint64_t i = 0; i < getSize(dlx); i++){
            y_data[i] = x[i] / scalar;
        }
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.abs")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t* x = static_cast<int32_t*>(dlx->data);
        for(uint64_t i = 0; i < getSize(dlx); i++){
            y_data[i] = std::abs(x[i]);
        }
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.max")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t* x = static_cast<int32_t*>(dlx->data);
        int max = x[0];
        for(uint64_t i = 1; i < getSize(dlx); i++){
            if(max < x[i]) max = x[i];
        }
        y_data[0] = max;
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.broadcast_max")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 4);
        DLTensor *a = args[0];
        DLTensor *b = args[1];
        DLTensor *c = args[2];
        int32_t *a_data = static_cast<int32_t*>(a->data);
        int32_t* b_data = static_cast<int32_t*>(b->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        if(b->ndim == 1){
#pragma omp parallel for
            for(uint64_t i = 0; i < getSize(a); i++){
                c_data[i] = a_data[i] > b_data[0] ? a_data[i] : b_data[0];
            }
        }else{
#pragma omp parallel for
        for(uint64_t i = 0; i < getSize(a); i++){
            uint64_t o_index = i;//broadcast_o_index(c->shape, c->ndim, o_index);
            uint64_t a_index = broadcast_i_index(c->shape, o_index, a->shape, a->ndim);
            uint64_t b_index = broadcast_i_index(c->shape, o_index, b->shape, b->ndim);
            //c_data[i] = (a_data[i] > b_data[i] ? a_data[i] : b_data[i]);
            c_data[i] = a_data[a_index] > b_data[b_index] ? a_data[a_index] : b_data[b_index];
        }
        }
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.concatenate")
.set_body([](CVMArgs args, CVMRetValue *ret){

#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
        int len = args.num_args;
        VERIFY(len >= 4);
        DLTensor *input0 = args[0];
        void *_attr = args[--len];
        auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
        auto &param = cvm::get<cvm::top::ConcatenateParam>(attr->parsed);
        DLTensor *out = args[--len];
        int32_t axis = param.axis;
        int32_t ndim = static_cast<int32_t>(input0->ndim);
        VERIFY(-ndim <= axis && axis < ndim);
        if(axis < 0) axis += ndim;
        VERIFY(axis < input0->ndim) << "axis out of bounds.";
        //TODO(kaihuo) check shape of all inputs
        int n_batch = input0->shape[0];
        // std::cerr << "n_batch " << n_batch << "\n";
        if (axis == 1 && n_batch == 1) {
            int32_t *out_data = static_cast<int32_t*>(out->data);
            uint64_t offset = 0;
            for(int k = 0; k < len; k++){
                DLTensor* input = args[k];
                int input_size_current = 1;
                //std::cerr << "\n";
                for (int i = 0; i < input->ndim; ++i) {
                    input_size_current *= input->shape[i];
                //    std::cerr << input->shape[i] << " " ;
                }
                //std::cerr << "\n";
                //std::cerr << "k = " << k << " " << input_size_current << "\n";
                memcpy(out_data + offset, input->data, sizeof(int32_t) * input_size_current);
                offset += input_size_current;
            }
        } else {
            int32_t *out_data = static_cast<int32_t*>(out->data);
            for(uint64_t i = 0; i < getSize(out); i++){
                uint64_t o_i = i, in_i = 0, in_i2 = 0, shapeSize = 0;
                for(int j = out->ndim-1; j >= 0; j--){
                    uint64_t col = o_i % out->shape[j];
                    o_i /= out->shape[j];
                    uint64_t tmpcol = col;
                    if(j == axis){
                        uint64_t allShapeSize = 0;
                        for(int k = 0; k < len; k++){
                            tmpcol = col - allShapeSize;
                            DLTensor *input = args[k];
                            allShapeSize += input->shape[axis];
                            if(col < allShapeSize){
                                in_i = k;
                                break;
                            }
                        }
                    }
                    in_i2 += (j == out->ndim-1 ? tmpcol : tmpcol * shapeSize);
                    DLTensor* input = args[in_i];
                    shapeSize = (j == out->ndim-1 ? input->shape[j] : shapeSize * input->shape[j]);
                }
                DLTensor *input = args[in_i];
                int32_t *input_data = static_cast<int32_t*>(input->data);
                out_data[i] = input_data[in_i2];
            }
        }
#ifdef CVM_PROFILING
        cvm_op_concat_cnt += omp_get_wtime() - start;
#endif
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.repeat")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    std::string str_axis = args[2];
    std::string str_repeat = args[3];
    int axis = std::atoi(str_axis.c_str());
    // int repeat = std::atoi(str_repeat.c_str());
    int ndim = x->ndim;
    if(axis < 0) axis = axis + ndim;

    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t o_i = i, in_i = 0, shapeSize = 0;
        for(int j = ndim-1; j >= 0; j--){
            int col = o_i % y->shape[j];
            o_i /= y->shape[j];
            int tmpcol = col;
            if(j == axis) tmpcol = col % x->shape[axis];
            in_i += (j == ndim-1 ? tmpcol : tmpcol * shapeSize);
            shapeSize = (j == ndim-1 ? x->shape[j] : shapeSize * x->shape[j]);
        }
        y_data[i] = x_data[in_i];
    }
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.negative")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    for(uint64_t i = 0; i < getSize(x); i++){
        y_data[i] = -x_data[i];
    }
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.slice_like")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
//    DLTensor *shape = args[1];
    DLTensor *y = args[2];
    std::string str_axis = args[3];
    int32_t *x_data = static_cast<int32_t*>(x->data);
    // TODO(kaihuo) check
//  int32_t *shape_like = static_cast<int32_t*>(shape->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int ndim = x->ndim;

   for(uint64_t i = 0; i < getSize(y); i++){
       uint64_t o_i = i, in_i = 0, shapeSize = 0;
       for(int j = ndim-1; j >= 0; j--){
           int col = o_i % y->shape[j];
           o_i /= y->shape[j];
           in_i += (j == ndim-1 ? col : col * shapeSize);
           shapeSize = (j == ndim-1 ? x->shape[j] : shapeSize * x->shape[j]);
       }
       y_data[i] = x_data[in_i];
   }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.tile")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    // DLTensor *repos = args[1];
    DLTensor *y = args[2];

    int32_t *x_data = static_cast<int32_t*>(x->data);
    // int32_t *repos_data = static_cast<int32_t*>(repos->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int32_t yndim = y->ndim;
    // TODO(kaihuo) check
    // int32_t rndim = repos->ndim;
    int32_t xndim = x->ndim;
    uint64_t tmp_y_size = 1;
    for(int i = 0; i < xndim; i++){
        tmp_y_size *= y->shape[i + yndim - xndim];
    }

    for(uint64_t i = 0; i < tmp_y_size; i++){
       uint64_t o_i = i, in_i = 0, shapeSize = 0;
       for(int j = xndim-1; j >= 0; j--){
            int yj = j + yndim - xndim;
            int col = o_i % y->shape[yj];
            o_i /= y->shape[yj];
            col = col % x->shape[j];
            in_i += (j == xndim-1 ? col : col * shapeSize);
            shapeSize = (j == xndim-1 ? x->shape[j] : shapeSize * x->shape[j]);
       }
       y_data[i] = x_data[in_i];
    }

    uint64_t othery = 1;
    for(int i = 0; i < yndim-xndim; i++){
        othery *= y->shape[i];
    }
    for(size_t i = 1; i < othery; i++){
        memcpy(y_data + i*tmp_y_size, y_data, tmp_y_size * sizeof(int32_t));
    }
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.expand_dims")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *ishape = args[0];
    int32_t axis = static_cast<int32_t>(args[1]);
    DLTensor *oshape = args[2];
    int32_t *ishape_data = static_cast<int32_t*>(ishape->data);
    int32_t *oshape_data = static_cast<int32_t*>(oshape->data);
    // TODO(kaihuo) check axis is -1
    for(uint64_t i = 0; i < getSize(oshape); i++){
        if(i < axis){
            oshape_data[i] = ishape_data[i];
        }else if(i == axis){
            oshape_data[i] = 1;
        }else{
            oshape_data[i] = ishape_data[i-1];
        }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.transpose")
.set_body([](CVMArgs args, CVMRetValue *ret){
    int num_args = args.num_args;
    VERIFY(num_args == 4 || num_args == 5);
    DLTensor *x = args[0];
    DLTensor *axes = nullptr; //args[1];
    DLTensor *y = nullptr; //args[2];
    if(num_args == 2){
        y = args[1];
    }else{
        axes = args[1];
        y = args[2];
    }

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t *axes_data = axes == nullptr ? nullptr : static_cast<int32_t*>(axes->data);
    int ndim = y->ndim;
    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t o_i = i, in_i = 0, shapeSize = 0;
        for(int j = ndim-1; j >= 0; j--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            int xj = j;//axes != nullptr ? axes[j] : j;
            if(axes != nullptr){
                xj = axes_data[j];
            }else{
                if(j == ndim-1) xj = 0;
                if(j == 0) xj = ndim-1;
            }
            int xi = 1;
            for(int tx = ndim-1; tx > xj; tx--){
                xi *= x->shape[tx];
            }
            in_i += col * xi;
        }
        y_data[i] = x_data[in_i];
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.slice_axis")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];

    int32_t axis;
    int32_t begin;
    int32_t end;

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int ndim = y->ndim;
    if(axis < 0) axis += ndim;
    if(begin < 0) begin += x->shape[axis];
    if(end < 0) end += x->shape[axis];

    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t o_i = i, in_i = 0, shapeSize = 0;
        for(int j = ndim-1; j >= 0; j--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            if (j == axis){
                col += begin;
            }
            in_i += (j == ndim-1 ? col : col * shapeSize);
            shapeSize = (j == ndim-1 ? x->shape[j] : shapeSize * x->shape[j]);
        }
        y_data[i] = x_data[in_i];
    }
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.slice")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *begin = args[1];
    DLTensor *end = args[2];
    DLTensor *step = args[3];
    DLTensor *y = args[4];

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t *begin_data = static_cast<int32_t*>(begin->data);
    int32_t *end_data = static_cast<int32_t*>(end->data);
    int32_t *step_data = static_cast<int32_t*>(step->data);

    int ndim = y->ndim;

    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t o_i = i, in_i = 0, shapeSize = 0;
        for(int j = ndim-1; j >= 0; j--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            col += (begin_data[j] < 0 ? begin_data[j] + x->shape[j] : begin_data[j]) + (step_data[j] < 0 ? step_data[j] + x->shape[j] : step_data[j]);
            col %= x->shape[j];
            in_i += (j == ndim-1 ? col : col * shapeSize);
            shapeSize = (j == ndim-1 ? x->shape[j] : shapeSize * x->shape[j]);
        }
        y_data[i] = x_data[in_i];
    }
});
/**
 * box_nms:
 */

#define FORMAT_CORNER 1
#define FORMAT_CENTER 2
uint32_t iou(const int32_t *rect1, const int32_t *rect2, const int32_t format){
    uint32_t x1_min = format == FORMAT_CORNER ? rect1[0] : rect1[0] - rect1[2]/2;
    uint32_t y1_min = format == FORMAT_CORNER ? rect1[1] : rect1[1] - rect1[3]/2;
    uint32_t x1_max = format == FORMAT_CORNER ? rect1[2] : x1_min + rect1[2];
    uint32_t y1_max = format == FORMAT_CORNER ? rect1[3] : y1_min + rect1[3];

    uint32_t x2_min = format == FORMAT_CORNER ? rect2[0] : rect2[0] - rect2[2]/2;
    uint32_t y2_min = format == FORMAT_CORNER ? rect2[1] : rect2[1] - rect2[3]/2;
    uint32_t x2_max = format == FORMAT_CORNER ? rect2[2] : x2_min + rect2[2];
    uint32_t y2_max = format == FORMAT_CORNER ? rect2[3] : y2_min + rect2[3];

    uint64_t sum_area = (x1_max-x1_min) * (y1_max-y1_min) + (x2_max-x2_min) * (y2_max-y2_min);

    if(x1_min > x2_max || x1_max < x2_min || y1_min > y2_max || y1_max < y2_min) return 0;
    uint32_t w = std::min(x1_max, x2_max) - std::max(x1_min, x2_min);
    uint32_t h = std::min(y1_max, y2_max) - std::max(y1_min, y2_min);
    uint64_t overlap_area = h*w;
    return static_cast<uint32_t>(overlap_area*100 / (sum_area - overlap_area));
}
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.box_nms")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void* _attr = args[2];

    int32_t overlap_thresh;
    int32_t valid_thresh;
    int32_t topk;
    int32_t coord_start;
    int32_t score_index;
    int32_t id_index;
    int32_t backgroud_id;
    int32_t force_suppress;
    int32_t in_format;
    int32_t out_format;

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int batch = 1;
    for(int i = 0; i < x->ndim - 2; i++){
        batch *= x->shape[i];
    }
    int n = x->shape[x->ndim-2];
    int k = x->shape[x->ndim-1];

    std::vector<int32_t*> rows(n);
    for (int i = 0; i < n; i++) {
        rows[i] = x_data + i * k;
    }
    std::sort(rows.begin(), rows.end(), [&score_index](const int32_t* a, const int32_t* b){
        return a[score_index] > b[score_index];
    });

    std::vector<bool> removed(n, false);
    int32_t y_index = 0;
    for(int i = 0; i < n; i++){
        int32_t *row1 = rows[i];
        if(row1[score_index] < valid_thresh) removed[i] = true;
        if(removed[i] == false){
            std::memcpy(&y_data[y_index], row1, k*sizeof(int32_t));
            y_index += k;
        }
        for(int j = i+1; j < n && !removed[i]; j++){
            int32_t* row2 = rows[j];
            if(iou(row1+coord_start, row2+coord_start, in_format) > overlap_thresh){
                removed[j] = true;
            }
        }
    }
    uint64_t ysize = getSize(y);
    if(y_index < ysize){
        std::memset(&y_data[y_index], -1, (ysize - y_index) * sizeof(int32_t));
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.get_valid_counts")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *x = args[0];
    DLTensor *valid_count = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];

    int32_t score_threshold; //TODO get from attr

    VERIFY(x->ndim == 3);
    int32_t batchs = x->shape[0];
    int32_t n = x->shape[1];
    int32_t k = x->shape[2];

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *valid_count_data = static_cast<int32_t*>(valid_count->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    for(int32_t i = 0; i < batchs; i++){
        int32_t y_index = 0;
        int32_t *input = x_data + i * n * k;
        int32_t *output = y_data + i * n * k;
        for(int32_t j = 0; j < n; j++){
            int32_t *row = input + j * k;
            if(row[1] > score_threshold){
                std::memcpy(&output[y_index * k], row, k * sizeof(int32_t));
                y_index += 1;
            }
        }
        valid_count_data[i] = y_index;
        if(y_index < n){
            std::memset(&output[y_index * k], -1, (n-y_index) * k * sizeof(int32_t));
        }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.non_max_suppression")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *x = args[0];
    DLTensor *valid_count = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];

    //TODO get from attr
    int32_t max_output_size;
    int32_t iou_threshold;
    int32_t topk;
    int32_t coord_start;
    int32_t score_index;
    int32_t id_index;
    bool force_suppress;
    bool return_indices;
    bool invalid_to_bottom;

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *valid_count_data = static_cast<int32_t*>(valid_count->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

   // int batch = 1;
   // for(int i = 0; i < x->ndim - 2; i++){
   //     batch *= x->shape[i];
   // }
   // int n = x->shape[x->ndim-2];
   // int k = x->shape[x->ndim-1];
    int32_t batchs = x->shape[0];
    int32_t n = x->shape[1];
    int32_t k = x->shape[2];

    for(int32_t b = 0; b < batchs; b++){
        int32_t vc = valid_count_data[b];
        std::vector<int32_t*> rows(n);
        int32_t *x_batch = x_data + b * n * k;
        int32_t *y_batch = y_data + b * n * k;

        for (int i = 0; i < n; i++) {
            rows[i] = x_batch + i * k;
        }
        std::sort(rows.begin(), rows.end(), [&score_index](const int32_t* a, const int32_t* b){
                return a[score_index] > b[score_index];
        });
        if(topk > 0 && topk < vc){
            for(int i = 0; i < vc - topk; i++){
                std::memset(rows[i+topk], -1, k * sizeof(int32_t));
            }
        }

        std::vector<bool> removed(n, false);
        for(int i = (topk < vc ? topk : vc); i < n; i++){
            removed[i] = true;
        }

        int32_t y_index = 0;
        for(int i = 0; i < vc; i++){
            int32_t *row1 = rows[i];
            if(removed[i] == false){
                std::memcpy(&y_batch[y_index*k], row1, k*sizeof(int32_t));
                y_index += 1;
            }
            for(int j = i+1; j < n && !removed[i] && iou_threshold > 0; j++){
                int32_t* row2 = rows[j];
                if(force_suppress || (id_index < 0 || row1[0] == row2[0])){
                    if(iou(row1+coord_start, row2+coord_start, FORMAT_CORNER) > iou_threshold){
                        removed[j] = true;
                    }
                }
            }
        }
        if(y_index < n){
            std::memset(&y_batch[y_index*k], -1, (n - y_index) * k * sizeof(int32_t));
        }
        if(max_output_size > 0){
            if(max_output_size < y_index){
                std::memset(&y_batch[max_output_size * k], -1, (y_index - max_output_size) * k * sizeof(int32_t));
            }
        }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.bias_add")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *x = args[0];
    DLTensor *bias = args[1];
    DLTensor *y = args[2];
    int32_t axis; //TODO get from attr
    int32_t ndim = x->ndim;
    VERIFY(axis > 0 && axis < ndim);

    const int32_t *x_data = static_cast<int32_t*>(x->data);
    const int32_t *bias_data = static_cast<int32_t*>(bias->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    for(uint64_t i = 0; i < getSize(y); i++){
        int32_t bV = 0;
        int64_t o_i = i;
        for(uint64_t j = ndim - 1; j >= 0; j--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            if(j == axis){
                bV = bias_data[axis];
                break;
            }
        }
        y_data[i] = x_data[i] + bV;
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.take")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *x = args[0];
    DLTensor *indices = args[1];
    DLTensor *y = args[2];
    DLTensor *_attr = args[3];

    int32_t axis = 0; //TODO get from attr

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *indices_data = static_cast<int32_t*>(indices->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int32_t yndim = y->ndim;
    int32_t xndim = x->ndim;
    int32_t indices_ndim = indices->ndim;

    for(uint64_t i = 0; i < getSize(y); i++){
        //y_data[i] = x_data[indices_data[i]];
        uint64_t o_i = i, x_i = 0, indices_i = 0, x_shape_size = 0, indices_shape_size = 0;
        for(uint32_t j = yndim - 1, k = indices_ndim-1; j>=axis; j--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            if(j < axis + indices_ndim){
                indices_i += (indices_shape_size == 0 ? col : col * indices_shape_size);
                indices_shape_size = (indices_shape_size == 0 ? indices->shape[k]
                        : indices_shape_size * indices->shape[k]);
                --k;
            }
        }

        o_i = i;
        int32_t k = xndim - 1;
        for(uint32_t j = yndim - 1; j >= axis + indices_ndim; j--, k--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            x_i += (j == yndim-1 ? col : col * x_shape_size);
            x_shape_size = (j == yndim-1 ? x->shape[k] : x_shape_size * x->shape[k]);
        }

        uint64_t x_indices_i = indices_data[indices_i];
        x_i += (x_shape_size == 0 ? x_indices_i : x_indices_i * x_shape_size);
        x_shape_size = (x_shape_size == 0 ? x->shape[k] : x_shape_size * x->shape[k]);
        --k;

        o_i = i;
        for(uint32_t j = yndim - 1; j>=0 && k >= 0; j--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            if(j < axis){
                x_i += x_shape_size == 0 ? col : col * x_shape_size;
                x_shape_size = x_shape_size == 0 ? x->shape[k] : x_shape_size * x->shape[k];
                --k;
            }
        }
        y_data[i] = x_data[x_i];
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.upsampling")
    .set_body([](CVMArgs args, CVMRetValue *ret){
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
    VERIFY(args.num_args == 3);
	DLTensor *x = args[0];
	DLTensor *y = args[1];

    VERIFY_EQ(x->ndim,     4) << "dimension should be 4D, Got: " << x->ndim;
    VERIFY_EQ(x->ndim,     y->ndim) << "dimension should match " << x->ndim << "!=" << y->ndim;
    VERIFY_EQ(x->shape[0], y->shape[0]) << "batch size should match";
    VERIFY_EQ(x->shape[1], y->shape[1]) << "batch size should match";

	void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::UpSamplingParam>(attr->parsed);
    VERIFY_EQ(param.method, "NEAREST_NEIGHBOR") << "only accept method = NEAREST_NEIGHBOR ";
    VERIFY_EQ(param.layout, "NCHW") << "only accept NHWC, Got:" << param.layout;

	int scale = {(int)param.scale};
    int h = x->shape[2], w = x->shape[3];
    int oh = y->shape[2], ow = y->shape[3];
    int n_batch = x->shape[0], n_channels = x->shape[1];

    auto x_data = static_cast<int32_t*>(x->data);
    auto y_data = static_cast<int32_t*>(y->data);

    // std::cerr << "scale = " << scale << "\n";
    // std::cerr << "input = " << x->shape[0] << " " << x->shape[1]
    //           << " " << x->shape[2] << " " << x->shape[3]
    //           << "\n";

    // std::cerr << "output = " << y->shape[0] << " " << y->shape[1]
    //           << " " << y->shape[2] << " " << y->shape[3]
    //           << "\n";

    //TODO(tian) optimize nested for-loop for scale
    #pragma omp parallel for collapse(2)
    for (uint32_t batch = 0; batch < n_batch; batch++) {
        for (uint32_t c = 0; c< n_channels; c++) {
            auto bc_y_data = y_data + batch * n_channels * oh * ow + c * oh * ow;
            auto bc_x_data = x_data + batch * n_channels *  h *  w + c *  h *  w;
            for (uint64_t xy = 0; xy < h * w; xy++) {
                uint32_t x = 2 * (xy / w), y = 2 * (xy % w);
                for (int xs = 0; xs < scale; xs++){
                    for (int ys = 0; ys < scale; ys++) {
                        bc_y_data[(x + xs) * ow + y + xs] = bc_x_data[xy];
                    }
                }
            }
        }
    }

#ifdef CVM_PROFILING
    cvm_op_upsampling_cnt += omp_get_wtime() - start;
    start = omp_get_wtime();
#endif

});


}
}

