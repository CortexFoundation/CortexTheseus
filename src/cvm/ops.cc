#include "graph_runtime.h"

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

#include "cuda_ops.h"
#include "omp.h"


namespace cvm {
namespace runtime {

#define CVM_RUNTIME_CUDA
#define DEBUG_OP false
inline void parseToIntPair(std::string str, int* ret){
    char a,b;
    sscanf(str.c_str(), "%c%d,%d%c", &a,ret, ret + 1, &b);
}

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
   for (uint64_t i = 0; i < getSize(x); i++) {
 		static_cast<int32_t*>(y->data)[i] = std::max(static_cast<int32_t*>(x->data)[i], 0);
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
  // assert(y->shape[0] == 1); // not tested yet
  for (uint32_t di = 0; di < y->shape[0]; di++) {
      for (uint32_t oi = 0; oi < y->shape[1]; oi++) {
          int32_t sum = 0;
          for (uint32_t xi = 0; xi < x->shape[1]; xi++) {
              sum += dx[di * x->shape[1] + xi] * dw[oi * w->shape[1] + xi];
          }
          if(db != nullptr){
              sum += db[oi];
          }
          dy[di * y->shape[1] + oi] = sum;
      }
  }

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
    int16_t int16[16];
    for(int i = 0; i < 16; i++) int16[i] = 1;
    __m256i vint16 = _mm256_loadu_si256((__m256i*)&int16);

    int blocks = K / 64 * 64;
    for(int i = 0; i < M; i++){
        int32_t bV = bias[i];
        for(int j = 0; j < N; j++){
            __m256i vc = _mm256_setzero_si256();
            int k = 0;
            for(k = 0; k < blocks; k+=32){
                __m256i va = _mm256_loadu_si256((__m256i*)&a[i*K+k]);
                __m256i vb = _mm256_loadu_si256((__m256i*)&tr_b[j*K+k]);
                __m256i vresult1 = _mm256_maddubs_epi16(vb, va);
                __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
                vc = _mm256_add_epi32(vresult2, vc);
            }
            int32_t sum = 0;
            for(int ti = 0; ti < 8; ti++){
                sum += ((int32_t*)&vc)[ti];
            }
            for(; k < K; k++){
                sum += a[i * K + k] * tr_b[j * K + k];
            }
            c[i*N+j] = sum + bV;
        }
    }

    free(tr_b);
}
void matrix_mul(const int8_t *a, const int8_t *b, const int32_t *bias,
        int32_t *c, const int M, const int K, const int N){
    std::memset(c, 0, sizeof(int32_t) * M * N);
#pragma omp parallel for
    for(int i = 0; i < M; i++){
        for(int k = 0; k < K; k++){
           int32_t aV = static_cast<int32_t>(a[i * K + k]);
            for(int j = 0; j < N; j++){
                c[i*N + j] += aV * static_cast<int32_t>(b[k*N + j]);
            }
        }
    }
    if (bias != NULL)
    for(int i = 0; i < M; i++){
        register int32_t biasV = bias[i];
        for(int j = 0; j < N; j++){
            c[i*N+j] += biasV;
        }
    }
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
    }
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
    int dlIndex = 2;
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
	int dilation[2] = {param.dilation[0], param.dilation[1]};
	int kernel_size[2] = {param.kernel_size[0], param.kernel_size[1]};
	int padding[2] = {param.padding[0], param.padding[1]};
	int strides[2] = {param.strides[0], param.strides[1]};

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
            flag = true;
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
    VERIFY(args.num_args == 3);
	DLTensor *x = args[0];
	DLTensor *y = args[1];
	void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::MaxPool2DParam>(attr->parsed);
	int strides[2] = {param.strides[0], param.strides[1]};
	int pool_size[2] = {param.pool_size[0], param.pool_size[1]};
	int padding[2] = {param.padding[0], param.padding[1]};

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

});

/*
* axis (2, 3)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.sum")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
		DLTensor *x = args[0];
		DLTensor *y = args[1];
        void *_attr = args[2];
        auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
        auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
		int axis[2] = {param.axis[0], param.axis[1]};
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

        for(uint64_t i = 0; i < getSize(args0); i++){
            c[i] = a[i] + b[i];
        }
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.reshape")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
		 void *_attr = args[2];
         auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
         auto &param = cvm::get<cvm::top::ReshapeParam>(attr->parsed);
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

         for(uint64_t i = 0; i < getSize(x); i++){
            y_data[i] = std::max(std::min(x_data[i], max), min);
         }
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

        for(uint64_t i = 0; i < getSize(a); i++){
            int32_t shift_a = a_data[i];
            if(b == 0)
                c_data[i] = shift_a;
            else{
                shift_a = ((a_data[i] >> (b-1)) +1) >> 1;
                c_data[i] = std::max(std::min(shift_a, max), min);
            }
        }

    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_left_shift")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
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
        int len = args.num_args;
        VERIFY(len >= 3);
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
    int repeat = std::atoi(str_repeat.c_str());
    int ndim = x->ndim;
    if(axis < 0) axis = axis + ndim;

    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t o_i = i, in_i = 0, shapeSize = 0;
        for(int j = ndim-1; j >= 0; j--){
            int col = o_i % y->shape[j];
            o_i /= y->shape[j];
            int tmpcol = col;
            if(j == axis) tmpcol = col / repeat;
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
//    int32_t *shape_like = static_cast<int32_t*>(shape->data);
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
/*********************************cuda op*********************************************/
#ifdef CVM_RUNTIME_CUDA
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.elemwise_add")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 3);
    DLTensor *a = args[0];
    DLTensor *b = args[1];
    DLTensor *c = args[2];
    int32_t *a_data = static_cast<int32_t*>(a->data);
    int32_t *b_data = static_cast<int32_t*>(b->data);
    int32_t *c_data = static_cast<int32_t*>(c->data);
    uint64_t n = getSize(a);
    const char *errorStr = cuda_elemwise_add(a_data, b_data, c_data, n, DEBUG_OP);
    VERIFY_EQ(errorStr == NULL, true) << errorStr;
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.conv2d")
.set_body([](CVMArgs args, CVMRetValue* rv){
    VERIFY(args.num_args == 13 || args.num_args == 12);
    DLTensor *x = args[0];
    DLTensor *w = args[1];
    int dlIndex = 2;
	DLTensor *b = nullptr;
    if(args.num_args == 13){
        b = args[dlIndex++];
    }
    DLTensor *y = args[dlIndex++];
	std::string groups_str = args[dlIndex++];
	std::string dilation_str = args[dlIndex++];
	std::string channels_str = args[dlIndex++];
	std::string layout_str = args[dlIndex++];
	std::string kernel_layout_str = args[dlIndex++];
	std::string kernel_size_str = args[dlIndex++];
	std::string padding_str = args[dlIndex++];
	std::string use_bias_str = args[dlIndex++];
	std::string strides_str = args[dlIndex++];
	int groups = std::atoi(groups_str.c_str());
	int dilation[2] = {0};
	parseToIntPair(dilation_str, dilation);
	//int channels = std::atoi(channels_str.c_str());
	int kernel_size[2] = {0};
	parseToIntPair(kernel_size_str, kernel_size);
	int padding[2] = {0};
	parseToIntPair(padding_str, padding);
	int strides[2] = {0};
	parseToIntPair(strides_str, strides);

    int32_t* x_data = (int32_t*)x->data;
    int32_t* w_data = (int32_t*)w->data;
    int32_t* y_data = (int32_t*)y->data;
	int32_t* b_data = b != nullptr ? (int32_t*)b->data : nullptr;

    int out_channels = static_cast<int>(w->shape[0]);
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
//  int o_h = static_cast<int>(y->shape[2]);
//  int o_w = static_cast<int>(y->shape[3]);

    if(groups == 1){
        const char* errorStr = cuda_conv2d(
                x_data, n_batch, in_channels, x_h, x_w,
                w_data, out_channels, in_channels, filter_h, filter_w,
                b_data,
                padding[0], padding[1],
                strides[0], strides[1],
                dilation[0], dilation[1],
                groups,
                y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, DEBUG_OP);
        VERIFY_EQ(errorStr == NULL, true) << errorStr;
    }else{
        const char* errorStr = cuda_depthwise_conv2d(
                x_data, n_batch, in_channels, x_h, x_w,
                w_data, out_channels, in_channels, filter_h, filter_w,
                b_data,
                padding[0], padding[1],
                strides[0], strides[1],
                dilation[0], dilation[1],
                groups,
                y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, DEBUG_OP);
        VERIFY_EQ(errorStr == NULL, true) << errorStr;

    }
 });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cuda_max_pool2d")
    .set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 6);
	DLTensor *x = args[0];
	DLTensor *y = args[1];
	std::string strides_str = args[2];
	std::string pool_size_str = args[3];
	std::string ceil_mode = args[4];
	std::string padding_str = args[5];
	int strides[2] = {0};
	parseToIntPair(strides_str, strides);
	int pool_size[2] = {0};
	parseToIntPair(pool_size_str, pool_size);
	int padding[2] = {0};
	parseToIntPair(padding_str, padding);

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

    const char* errorStr = cuda_max_pool(
            x_data, n_batch, in_channels, x_h, x_w,
            filter_h, filter_w,
            padding[0], padding[1],
            strides[0], strides[1],
            y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, DEBUG_OP);
    VERIFY_EQ(errorStr == NULL, true) << errorStr;
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.dense")
.set_body([](CVMArgs args, CVMRetValue* rv) {
        VERIFY(args.num_args == 6 || args.num_args == 5);
        int ndim = args.num_args;
        DLTensor *x = args[0];
        DLTensor *w = args[1];
        DLTensor *b = nullptr;
        DLTensor *y = nullptr;
        int32_t* db = nullptr;
        if(ndim == 6){
            b = args[2];
            y = args[3];
            db = static_cast<int32_t*>(b->data);
        }else{
            y = args[2];
        }
        auto dx = static_cast<int32_t*>(x->data);
        auto dy = static_cast<int32_t*>(y->data);
        auto dw = static_cast<int32_t*>(w->data);

        const char* errorStr = cuda_dense(
                dx, dw, dy,
                static_cast<int32_t>(x->shape[0]),
                static_cast<int32_t>(x->shape[1]),
                static_cast<int32_t>(y->shape[1]),
                db,
                DEBUG_OP);
        VERIFY_EQ(errorStr == NULL, true) << errorStr;
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.clip").set_body([](CVMArgs args, CVMRetValue* rv) {
   VERIFY(args.num_args == 4);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   std::string min_str = args[2];
   std::string max_str = args[3];
   int min = std::atoi(min_str.c_str());
   int max = std::atoi(max_str.c_str());

   const char *errorStr = cuda_clip(
           static_cast<int32_t*>(x->data),
           static_cast<int32_t*>(y->data),
           getSize(x),
           max, min, DEBUG_OP);
   VERIFY_EQ(errorStr == NULL, true) << errorStr;
 });

 CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.relu").set_body([](CVMArgs args, CVMRetValue* rv) {
   VERIFY(args.num_args == 2);
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   const char* errorStr = cuda_relu(
           static_cast<int32_t*>(x->data),
           static_cast<int32_t*>(y->data),
           getSize(x),
           DEBUG_OP);
    VERIFY_EQ(errorStr == NULL, true) << errorStr;
 });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.flatten").set_body([]
(CVMArgs args, CVMRetValue* rv){
     VERIFY(args.num_args == 2);
     DLTensor *x = args[0];
     DLTensor *y = args[1];

     const char* errorStr = cuda_flatten(
            static_cast<int32_t*>(x->data),
            static_cast<int32_t*>(y->data),
            getSize(x),
            DEBUG_OP);
     VERIFY_EQ(errorStr == NULL, true) << errorStr;
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_add")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        int64_t *ashape = static_cast<int64_t*>(args0->shape);
        int32_t adim = static_cast<int32_t>(args0->ndim);
        int64_t *bshape = static_cast<int64_t*>(args1->shape);
        int32_t bdim = static_cast<int32_t>(args1->ndim);
        int64_t *cshape = static_cast<int64_t*>(args2->shape);
        int32_t cdim = static_cast<int32_t>(args2->ndim);


        const char* errorStr = cuda_broadcast_add(a, b, c, getSize(args0),
		ashape, adim,
		bshape, bdim,
		cshape, cdim, DEBUG_OP);
        VERIFY_EQ(errorStr == NULL, true) << errorStr;
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_sub")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        int64_t *ashape = static_cast<int64_t*>(args0->shape);
        int32_t adim = static_cast<int32_t>(args0->ndim);
        int64_t *bshape = static_cast<int64_t*>(args1->shape);
        int32_t bdim = static_cast<int32_t>(args1->ndim);
        int64_t *cshape = static_cast<int64_t*>(args2->shape);
        int32_t cdim = static_cast<int32_t>(args2->ndim);


        const char* errorStr = cuda_broadcast_sub(a, b, c, getSize(args0),
		ashape, adim,
		bshape, bdim,
		cshape, cdim, DEBUG_OP);

        VERIFY_EQ(errorStr == NULL, true) << errorStr;
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_mul")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        int64_t *ashape = static_cast<int64_t*>(args0->shape);
        int32_t adim = static_cast<int32_t>(args0->ndim);
        int64_t *bshape = static_cast<int64_t*>(args1->shape);
        int32_t bdim = static_cast<int32_t>(args1->ndim);
        int64_t *cshape = static_cast<int64_t*>(args2->shape);
        int32_t cdim = static_cast<int32_t>(args2->ndim);


        const char* errorStr = cuda_broadcast_mul(a, b, c, getSize(args0),
		ashape, adim,
		bshape, bdim,
		cshape, cdim, DEBUG_OP);

        VERIFY_EQ(errorStr == NULL, true) << errorStr;
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_div")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        int64_t *ashape = static_cast<int64_t*>(args0->shape);
        int32_t adim = static_cast<int32_t>(args0->ndim);
        int64_t *bshape = static_cast<int64_t*>(args1->shape);
        int32_t bdim = static_cast<int32_t>(args1->ndim);
        int64_t *cshape = static_cast<int64_t*>(args2->shape);
        int32_t cdim = static_cast<int32_t>(args2->ndim);


        const char* errorStr = cuda_broadcast_div(a, b, c, getSize(args0),
		ashape, adim,
		bshape, bdim,
		cshape, cdim, DEBUG_OP);

        VERIFY_EQ(errorStr == NULL, true) << errorStr;
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_right_shift")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        int64_t *ashape = static_cast<int64_t*>(args0->shape);
        int32_t adim = static_cast<int32_t>(args0->ndim);
        int64_t *bshape = static_cast<int64_t*>(args1->shape);
        int32_t bdim = static_cast<int32_t>(args1->ndim);
        int64_t *cshape = static_cast<int64_t*>(args2->shape);
        int32_t cdim = static_cast<int32_t>(args2->ndim);

        const char* errorStr = cuda_broadcast_right_shift(a, b, c, getSize(args0),
		ashape, adim,
		bshape, bdim,
		cshape, cdim, DEBUG_OP);

        VERIFY_EQ(errorStr == NULL, true) << errorStr;
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_left_shift")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *args0 = args[0];
        DLTensor *args1 = args[1];
        DLTensor *args2 = args[2];
        int32_t *a = static_cast<int32_t*>(args0->data);
        int32_t *b = static_cast<int32_t*>(args1->data);
        int32_t *c = static_cast<int32_t*>(args2->data);
        int64_t *ashape = static_cast<int64_t*>(args0->shape);
        int32_t adim = static_cast<int32_t>(args0->ndim);
        int64_t *bshape = static_cast<int64_t*>(args1->shape);
        int32_t bdim = static_cast<int32_t>(args1->ndim);
        int64_t *cshape = static_cast<int64_t*>(args2->shape);
        int32_t cdim = static_cast<int32_t>(args2->ndim);

        const char* errorStr = cuda_broadcast_left_shift(a, b, c, getSize(args0),
		ashape, adim,
		bshape, bdim,
		cshape, cdim, DEBUG_OP);

        VERIFY_EQ(errorStr == NULL, true) << errorStr;
    });

/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.max_pool2d")
    .set_body([](CVMArgs args, CVMRetValue *ret){
            VERIFY(args.num_args == 6);
            DLTensor *x = args[0];
            DLTensor *y = args[1];
            std::string strides_str = args[2];
            std::string pool_size_str = args[3];
            std::string ceil_mode = args[4];
            std::string padding_str = args[5];
            int strides[2] = {0};
            parseToIntPair(strides_str, strides);
            int pool_size[2] = {0};
            parseToIntPair(pool_size_str, pool_size);
            int padding[2] = {0};
            parseToIntPair(padding_str, padding);

            int32_t* x_data = (int32_t*)x->data;
            int32_t* y_data = (int32_t*)y->data;

            int filter_h = pool_size[0];
            int filter_w = pool_size[1];

            int n_batch = static_cast<int>(x->shape[0]);
            int in_channels = static_cast<int>(x->shape[1]);
            int out_channels = in_channels;
            int x_h = static_cast<int>(x->shape[2]);
            int x_w = static_cast<int>(x->shape[3]);
            //	int o_h = (x_h + 2 * padding[0] - filter_h) / strides[0] + 1;
            //	int o_w = (x_w + 2 * padding[1] - filter_w) / strides[1] + 1;
            int o_h = static_cast<int>(y->shape[2]);
            int o_w = static_cast<int>(y->shape[3]);
            const char* errorStr = cuda_max_pool(
                    x_data, n_batch, in_channels, x_h, x_w,
                    filter_h, filter_w,
                    padding[0], padding[1],
                    strides[0], strides[1],
                    y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, DEBUG_OP);
            VERIFY_EQ(errorStr == NULL, true) << errorStr;
});

/*
* axis (2, 3)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.sum")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
		DLTensor *x = args[0];
		DLTensor *y = args[1];
		std::string axis_str = args[2];
		int axis[2] = {0};
		parseToIntPair(axis_str, axis);

		int32_t *x_data = static_cast<int32_t*>(x->data);
		int32_t *y_data = static_cast<int32_t*>(y->data);
		int n_batch = static_cast<int>(x->shape[0]);
		int channels = static_cast<int>(x->shape[1]);
		int x_h = static_cast<int>(x->shape[2]);
		int x_w = static_cast<int>(x->shape[3]);
        const char* errorStr = cuda_sum(x_data, n_batch, channels, x_h, x_w, y_data, DEBUG_OP);
        VERIFY_EQ(errorStr == NULL, true) << errorStr;
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.reshape")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
         DLTensor *x = args[0];
		 DLTensor *y = args[1];
         std::string newshape = args[2];
         const char* errorStr = cuda_reshape(
                 static_cast<int32_t*>(x->data),
                 static_cast<int32_t*>(y->data),
                 getSize(x),
                 DEBUG_OP);
         VERIFY_EQ(errorStr == NULL, true) << errorStr;
    });
/*\brief:
 * x, input data
 * y, output data
 * precision, clip precision
 */
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.cvm_clip")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
         DLTensor *x = args[0];
     DLTensor *y = args[1];
         int32_t *x_data = static_cast<int32_t*>(x->data);
         int32_t *y_data = static_cast<int32_t*>(y->data);
         std::string str_precision = args[2];
         int32_t precision = std::atoi(str_precision.c_str());
         VERIFY(precision > 0) << "precision must greater zero";
         const char* errorStr = cuda_cvm_clip(
                 x_data,
                 precision,
                 y_data,
                 getSize(x),
                 DEBUG_OP);
         VERIFY(errorStr == NULL) << errorStr;
    });

/*
 * a, input data
 * c, output data
 * precision, clip precision
 * b, shift b
 * */
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.cvm_right_shift")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 4);
        DLTensor *a = args[0];
        DLTensor *c = args[1];
        std::string str_precision = args[2];
        std::string str_b = args[3];
        int32_t precision = std::atoi(str_precision.c_str());
        int32_t b = std::atoi(str_b.c_str());
        int32_t* a_data = static_cast<int32_t*>(a->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        VERIFY_GT(precision, 0) << "precision must greater zero";
        const char* errorStr = cuda_cvm_right_shift(
                a_data,
                b,
                precision,
                c_data,
                getSize(a),
                DEBUG_OP);
        VERIFY(errorStr == NULL) << errorStr;

    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.cvm_left_shift")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 4);
        DLTensor *a = args[0];
        DLTensor *c = args[1];
        std::string str_precision = args[2];
        std::string str_b = args[3];
        int32_t precision = std::atoi(str_precision.c_str());
        int32_t b = std::atoi(str_b.c_str());
        int32_t* a_data = static_cast<int32_t*>(a->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        VERIFY_GT(precision, 0) << "precision must greater zero";
        const char* errorStr = cuda_cvm_left_shift(
                a_data,
                b,
                precision,
                c_data,
                getSize(a),
                DEBUG_OP);
        VERIFY(errorStr == NULL) << errorStr;
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.log2")
    .set_body([](CVMArgs args, CVMRetValue *ret){
//        std::string x_str = args[0];
        VERIFY(args.num_args == 2);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t *x = static_cast<int32_t*>(dlx->data);
        const char* errorStr = cuda_log(x, y_data, DEBUG_OP);
        VERIFY(errorStr == NULL) << errorStr;
    });
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.abs")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 2);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t* x = static_cast<int32_t*>(dlx->data);
        const char* errorStr = cuda_abs(x, y_data, getSize(dlx), DEBUG_OP);
        VERIFY(errorStr == NULL) << errorStr;
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.max")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 2);
        DLTensor *dlx = args[0];
        DLTensor *y = args[1];
        int32_t *y_data = static_cast<int32_t*>(y->data);
        int32_t* x = static_cast<int32_t*>(dlx->data);
        const char* errorStr = cuda_max(x, y_data, getSize(dlx), DEBUG_OP);
        VERIFY(errorStr == NULL) << errorStr;
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_max")
    .set_body([](CVMArgs args, CVMRetValue *ret){
        VERIFY(args.num_args == 3);
        DLTensor *a = args[0];
        DLTensor *b = args[1];
        DLTensor *c = args[2];
        int32_t *a_data = static_cast<int32_t*>(a->data);
        int32_t* b_data = static_cast<int32_t*>(b->data);
        int32_t* c_data = static_cast<int32_t*>(c->data);
        int64_t *ashape = static_cast<int64_t*>(a->shape);
        int32_t adim = static_cast<int32_t>(a->ndim);
        int64_t *bshape = static_cast<int64_t*>(b->shape);
        int32_t bdim = static_cast<int32_t>(b->ndim);
        int64_t *cshape = static_cast<int64_t*>(c->shape);
        int32_t cdim = static_cast<int32_t>(c->ndim);

        const char* errorStr = cuda_broadcast_max(a_data, b_data, c_data, getSize(a),
		ashape, adim,
		bshape, bdim,
		cshape, cdim, DEBUG_OP);
        VERIFY(errorStr == NULL) << errorStr;
    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.concatenate")
.set_body([](CVMArgs args, CVMRetValue *ret){
        int len = args.num_args;
        VERIFY(len >= 3);
        DLTensor *input0 = args[0];
        std::string str_axis = args[--len];
        DLTensor *output = args[--len];
        int32_t axis = std::atoi(str_axis.c_str());
        int32_t ndim = static_cast<int32_t>(input0->ndim);
        VERIFY(-ndim <= axis && axis < ndim);
        if(axis < 0) axis += ndim;
        VERIFY(axis < input0->ndim) << "axis out of bounds.";

        int32_t *out_data = static_cast<int32_t*>(output->data);
        int64_t preSize = 0;
        for(int i = 0; i < len; i++){
            DLTensor *input  = args[i];
            const char* errorStr = cuda_concatenate(
                    static_cast<int32_t*>(input->data),
                    input->shape,
                    input->ndim,
                    getSize(input),
                    out_data,
                    output->shape,
                    output->ndim,
                    getSize(output),
                    preSize,
                    preSize + input->shape[axis],
                    axis,
                    DEBUG_OP
            );
            VERIFY(errorStr == NULL) << errorStr;
            preSize += input->shape[axis];
        }
});
#endif // end of CVM_RUNTIME_CUDA
}
}

