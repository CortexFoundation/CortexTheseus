#include "ops.h"

#include "omp.h"
#include <immintrin.h>

#define CVM_PROFILING

namespace cvm {
namespace runtime {

double transpose_int8_avx256_transpose_cnt = 0;
double transpose_int8_avx256_gemm_cnt = 0;
double im2col_cnt = 0;
double cvm_op_dense_cnt = 0;
double cvm_op_maxpool_cnt = 0;
double cvm_op_concat_cnt = 0;
double cvm_op_upsampling_cnt = 0;
double cvm_op_inline_matmul_cnt = 0;
double cvm_op_elemwise_cnt = 0;
double cvm_op_chnwise_conv_cnt = 0;
double cvm_op_chnwise_conv1x1_cnt = 0;
double cvm_op_depthwise_conv_cnt = 0;
double cvm_op_depthwise_conv1x1_cnt = 0;

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.relu")
.set_body([](CVMArgs args, CVMRetValue* rv){
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   int32_t *x_data = static_cast<int32_t*>(x->data);
   int32_t *y_data = static_cast<int32_t*>(y->data);
#pragma omp parallel for
   for (uint64_t i = 0; i < getSize(x); i++) {
        auto tmp = x_data[i];
        if (tmp < 0) tmp = 0;
        y_data[i] = tmp;
   }
#ifdef CVM_PROFILING
    cvm_op_elemwise_cnt += omp_get_wtime() - start;
#endif
  print_to_file(y, "relu.txt");
});

/*
* x : M*K
* w : N*K
* b : N
* y : M*N
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.dense")
.set_body([](CVMArgs args, CVMRetValue* rv) {
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
  int ndim = args.num_args;
  DLTensor *x = args[0];
  DLTensor *w = args[1];
  DLTensor *bias = nullptr;
  DLTensor *y = nullptr;
  int32_t* bias_data = nullptr;
  if(ndim == 5){
    bias = args[2];
    y = args[3];
    bias_data = static_cast<int32_t*>(bias->data);
  } else{
    y = args[2];
  }

  auto x_data = static_cast<int32_t*>(x->data);
  auto y_data = static_cast<int32_t*>(y->data);
  auto w_data = static_cast<int32_t*>(w->data);
#pragma omp parallel for
  for (int64_t di = 0; di < y->shape[0]; ++di) {
    int32_t y_offset = di * y->shape[1], x_offset = di * x->shape[1];
    for (int64_t oi = 0; oi < y->shape[1]; ++oi) {
      int32_t sum = 0, w_offset = oi * w->shape[1];
      for (int64_t xi = 0; xi < x->shape[1]; ++xi) {
        sum += x_data[x_offset + xi] * w_data[w_offset + xi];
      }
      y_data[y_offset + oi] = sum;
    }
  }
  if (bias_data != nullptr) {
#pragma omp parallel for
    for (int64_t di = 0; di < y->shape[0]; ++di) {
      int32_t y_offset = di * y->shape[1];
      for (int64_t oi = 0; oi < y->shape[1]; ++oi) {
        y_data[y_offset + oi] += bias_data[oi];
      }
    }
  }
  print_to_file(y, "dense.txt");

#ifdef CVM_PROFILING
  cvm_op_dense_cnt += omp_get_wtime() - start;
#endif
});

void transpose_int8_avx256(const int8_t *a, const int8_t *b, const int32_t *bias,
    int32_t *c, const int M, const int K, const int N){
  std::shared_ptr<int8_t> tr_b(new int8_t[K*N]);

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
          tr_b.get()[(j+ti) * K + i + tj] = tile[ti][tj];
        }
      }
    }
    for(int ti = 0; ti < 32; ti++){
      for(int tj = j; tj < N; tj++){
        tr_b.get()[tj * K + i+ti] = b[(i+ti) * N + tj];
      }
    }
  }
  for(; i < K; i++){
    for(j = 0; j < N; j++){
      tr_b.get()[j * K + i] = b[i * N + j];
    }
  }
  int16_t int16[16];
  for(int i = 0; i < 16; i++) int16[i] = 1;
  __m256i vint16 = _mm256_loadu_si256((__m256i*)&int16);

  int blocks = K / 32 * 32;
  for(int i = 0; i < M; i++){
    int32_t bV = bias != NULL ? bias[i] : 0;
    for(int j = 0; j < N; j++){
      __m256i vc = _mm256_setzero_si256();
      int k = 0;
      for(k = 0; k < blocks; k+=32){
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i*K+k]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&tr_b.get()[j*K+k]);
        __m256i vresult1 = _mm256_maddubs_epi16(vb, va);
        __m256i vresult2 = _mm256_madd_epi16(vresult1, vint16);
        vc = _mm256_add_epi32(vresult2, vc);
      }
      int32_t sum = 0;
      for(int ti = 0; ti < 8; ti++){
        sum += ((int32_t*)&vc)[ti];
      }
      for(; k < K; k++){
        sum += a[i * K + k] * tr_b.get()[j * K + k];
      }
      c[i*N+j] = sum + bV;
    }
  }
}

void transpose(const int8_t *A, int8_t *B, int K, int N) {
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < K; k++) {
            B[i * K + k] = A[k * N + i];
        }
    }
}

void matrix_mul(const int8_t *a, const int8_t *b, const int32_t *bias,
        int32_t *c, const int M, const int K, const int N, int algo = 0)
{
#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
    if(std::memset(c, 0, sizeof(int32_t) * M * N) == NULL){
      CHECK(false);
    }

    if (N > M ) {
#pragma omp parallel for
        for(int i = 0; i < M; i++){
            for(int k = 0; k < K; k++){
                int32_t aV = static_cast<int32_t>(a[i * K + k]);
                for(int j = 0; j < N; j++){
                    c[i*N + j] += aV * static_cast<int32_t>(b[k*N + j]);
                }
            }
        }
    } else {
      std::vector<int8_t> tr_b;
      try{
        tr_b.resize(N*K);
      }catch(const std::bad_alloc& e){
        CHECK(false);
      }

      transpose(b, tr_b.data(), K, N);
      #pragma omp parallel
      {
          int i, j, k;
          #pragma omp for
          for (i = 0; i < M; i++) {
              auto ap = a + i * K;
              for (j = 0; j < N; j++) {
                  int32_t dot = 0;
                  auto tr_bp = tr_b.data() + j * K;
                  for (k = 0; k < K; k++) {
                      dot += ap[k] * static_cast<int32_t>(tr_bp[k]);
                  }
                  c[i*N + j] = dot;
              }
          }
      }
    }

    if(bias != NULL){
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < M; i++){
            for(int j = 0; j < N; j++){
                c[i*N+j] += bias[i];
            }
        }
    }
#ifdef CVM_PROFILING
    double cost_time = omp_get_wtime() - start;
    // std::cerr << "matrix_mul = " << M << " " << K << " " << N << " " << M * K * N << "  " << cost_time << "\n";
    cvm_op_inline_matmul_cnt += cost_time;
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
    int8_t* data_col, bool &has_negetive)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
  // auto data_col_init = data_col;
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
                if(tv < 0) {
                  has_negetive = true;
                }
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
#ifdef CVM_PROFILING
  im2col_cnt +=  omp_get_wtime() - start;
#endif
}

void depthwise_conv2d(
   int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
   int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
   int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
   int32_t *b_data,
   int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
   int32_t groups)
{
  // TODO(kaihuo) optimize cpu's depthwise conv efficiency, e.g. using modified im2col
  for(int32_t n = 0; n < n_batch; ++n){
    for(int32_t c = 0; c < in_channels; ++c){
      for(int32_t h = 0; h < o_h; ++h){
        for(int32_t w = 0; w < o_w; ++w){
          int32_t sum = 0;
          for(int32_t fh = 0; fh < filter_h; ++fh){
            for(int32_t fw = 0; fw < filter_w; ++fw){
                int32_t th = h * stride_h + fh*dilation_h - padding[0];
                int32_t tw = w * stride_w + fw*dilation_w - padding[1];
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

void depthwise_conv2d_single(
   int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
   int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
   int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
   int32_t *b_data,
   int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
   int32_t groups)
{
  // std::cerr << "depth wise imcol\n";
  int32_t fn = out_channels * filter_h * filter_w;
  std::shared_ptr<int8_t> int8_filter(new int8_t[fn]);
  for(int32_t i = 0; i < fn; i++){
    int8_filter.get()[i] = static_cast<int8_t>(w_data[i]);
  }
  const int M = 1;
  const int K = filter_h * filter_w;
  const int N = o_h * o_w;

  std::shared_ptr<int8_t> data_col(new int8_t[in_channels * filter_h * filter_w * o_h * o_w]);
  bool has_negetive = false;
  im2col_cpu(
    x_data + 0* in_channels * x_h * x_w, //+ channel * x_h * x_w,
    in_channels, x_h, x_w,
    filter_h, filter_w,
    padding[0], padding[1],
    stride_h, stride_w,
    dilation_h, dilation_w,
    data_col.get(), has_negetive
  );
  if(std::memset(y_data, 0, sizeof(int32_t) * in_channels * M * N) == NULL){
    CHECK(false);
  }
  for(int batch = 0; batch < n_batch; batch++) {
    auto y_data_batch = y_data + batch * in_channels * N;
    #pragma omp parallel for
    for (int channel = 0; channel < out_channels; channel++) {
      auto c = y_data_batch + channel * N;
      auto a = int8_filter.get() + channel * K;
      auto b = data_col.get() + channel * K * N;
      for(int k = 0; k < K; k++){
        int32_t aV = static_cast<int32_t>(a[k]);
        for(int j = 0; j < N; j++){
          c[j] += aV * static_cast<int32_t>(b[k*N + j]);
        }
      }
      if (b_data) {
        for(int j = 0; j < N; j++){
          c[j] += b_data[channel];
        }
      }
    }
  }
}

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.conv2d")
    .set_body([](CVMArgs args, CVMRetValue* rv)
{
  DLTensor *x = args[0];
  DLTensor *w = args[1];
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
  int t_filter_h = (filter_h - 1) * dilation[0] + 1;
  int t_filter_w = (filter_w - 1) * dilation[1] + 1;

  int n_batch = static_cast<int>(x->shape[0]);
  int in_channels = static_cast<int>(x->shape[1]);
  int x_h = static_cast<int>(x->shape[2]);
  int x_w = static_cast<int>(x->shape[3]);
  int o_h = (x_h + 2 * padding[0] - t_filter_h) / strides[0] + 1;
  int o_w = (x_w + 2 * padding[1] - t_filter_w) / strides[1] + 1;

  if(groups > 1){
#ifdef CVM_PROFILING
        double start = omp_get_wtime();
#endif
    depthwise_conv2d(
        x_data, n_batch, in_channels, x_h, x_w,
        w_data, filter_c, filter_h, filter_w,
        y_data, out_channels, o_h, o_w,
        b_data,
        padding, stride_h, stride_w, dilation[0], dilation[1],
        groups);
#ifdef CVM_PROFILING
    cvm_op_depthwise_conv_cnt += omp_get_wtime() - start;
#endif
    } else {
#ifdef CVM_PROFILING
      double start = omp_get_wtime();
      //double start_1x1 = omp_get_wtime();
#endif
      std::shared_ptr<int8_t> data_col(new int8_t[in_channels * filter_h * filter_w * o_h * o_w]);
      int32_t fn = out_channels * in_channels * filter_h * filter_w;
      std::shared_ptr<int8_t> int8_filter(new int8_t[fn]);

      for(int32_t i = 0; i < fn; i++){
          int8_filter.get()[i] = static_cast<int8_t>(w_data[i]);
      }
      for(int32_t i = 0; i < n_batch; i++){
          bool has_negetive = false;
          im2col_cpu(x_data + i * in_channels * x_h * x_w, in_channels, x_h, x_w, filter_h, filter_w, padding[0], padding[1],
                  stride_h, stride_w, dilation_h, dilation_w, data_col.get(), has_negetive);
          const int32_t M = out_channels;
          const int32_t K = in_channels * filter_h * filter_w;
          const int32_t N = o_h * o_w;
          if(has_negetive) {
            matrix_mul(int8_filter.get(), data_col.get(), b_data, y_data + i * out_channels * o_h * o_w,
                  M, K, N);
          }else{
            transpose_int8_avx256(int8_filter.get(), data_col.get(), b_data, y_data + i * out_channels * o_h * o_w,
                  M, K, N);
          }
      }
#ifdef CVM_PROFILING
        cvm_op_chnwise_conv_cnt += omp_get_wtime() - start;
        if (filter_h == 1 && filter_w == 1) {
          cvm_op_chnwise_conv1x1_cnt += omp_get_wtime() - start;
        }
#endif
    }
  print_to_file(y, "conv2d.txt");
});



/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm.max_pool2d")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
  DLTensor *x = args[0];
  DLTensor *y = args[1];
  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::MaxPool2DParam>(attr->parsed);
  int padding[2] = {(int)param.padding[0], (int)param.padding[0]};
  if(param.padding.ndim() == 2){
    padding[1] = (int)param.padding[1];
  }

  int stride_h = param.strides[0];
  int stride_w = param.strides[1];

  int32_t* x_data = (int32_t*)x->data;
  int32_t* y_data = (int32_t*)y->data;

  int filter_h = param.pool_size[0];
  int filter_w = param.pool_size[1];

  int n_batch = static_cast<int>(x->shape[0]);
  int in_channels = static_cast<int>(x->shape[1]);
  int out_channels = in_channels;
  int x_h = static_cast<int>(x->shape[2]);
  int x_w = static_cast<int>(x->shape[3]);
  int o_h = static_cast<int>(y->shape[2]);
  int o_w = static_cast<int>(y->shape[3]);
#define GETX(n, c, h, w) x_data[(n) * in_channels * x_h * x_w + (c) * x_h * x_w + (h) * x_w + (w)]
#define GETY(n, c, h, w) y_data[(n) * out_channels * o_h * o_w + (c) * o_h * o_w + (h) * o_w + (w)]
  auto calc_func = [&](int n, int k, int p, int q) {
    int32_t y_max = int32_t(1)<<31;
    for (int r = 0; r < filter_h; ++r) {
      for (int s = 0; s < filter_w; ++s) {
        int32_t tp = p * stride_h + r - padding[0];
        int32_t tq = q * stride_w + s - padding[1];
        int32_t x_tmp = 0; // zero padding by default
        if (0 <= tp && tp < x_h && 0 <= tq && tq < x_w)
          x_tmp = GETX(n, k, tp, tq);
        y_max = std::max(x_tmp, y_max);
      }
    }
    return y_max;
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

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_precision")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t *x_data = static_cast<int32_t*>(x->data);
    for(size_t j = 0; j < getSize(x); j++){
      int64_t x_val = x_data[j];
      y_data[j] = 64;
      for(int i = 1; i < 64; i++){
        int64_t tmp = (int64_t)1 << i;
        if(std::abs(x_val) < tmp){
          y_data[j] = i;
          break;
        }
      }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.abs")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t* x_data = static_cast<int32_t*>(x->data);
    for(uint64_t i = 0; i < getSize(x); i++){
      y_data[i] = std::abs(x_data[i]);
    }
});


CVM_REGISTER_GLOBAL("cvm.runtime.cvm.concatenate")
.set_body([](CVMArgs args, CVMRetValue *ret){
#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
    int len = args.num_args;
    DLTensor *input0 = args[0];
    void *_attr = args[--len];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::ConcatenateParam>(attr->parsed);
    DLTensor *out = args[--len];
    int32_t axis = param.axis;
    int32_t ndim = static_cast<int32_t>(input0->ndim);
    if(axis < 0) axis += ndim;
    int n_batch = input0->shape[0];

    if (axis == 1 && n_batch == 1) {
      int32_t *out_data = static_cast<int32_t*>(out->data);
      uint64_t offset = 0;
      for(int k = 0; k < len; k++){
        DLTensor* input = args[k];
        int input_size_current = 1;
        for (int i = 0; i < input->ndim; ++i) {
          input_size_current *= input->shape[i];
        }
        memcpy(out_data + offset, input->data, sizeof(int32_t) * input_size_current);
        offset += input_size_current;
      }
    } else {
      int32_t *out_data = static_cast<int32_t*>(out->data);
      for(uint64_t i = 0; i < getSize(out); i++){
        uint64_t o_i = i, in_i = 0, in_i2 = 0, shapeSize = 1;
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
          in_i2 += tmpcol * shapeSize; 
          DLTensor* input = args[in_i];
          shapeSize *= input->shape[j];
        }
        DLTensor *input = args[in_i];
        int32_t *input_data = static_cast<int32_t*>(input->data);
        out_data[i] = input_data[in_i2];
      }
    }
#ifdef CVM_PROFILING
    cvm_op_concat_cnt += omp_get_wtime() - start;
#endif
  print_to_file(out, "concatenate.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.repeat")
.set_body([](CVMArgs args, CVMRetValue *ret){
#ifdef CVM_PROFILING
    double start = omp_get_wtime();
#endif
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::RepeatParam>(attr->parsed);
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t axis = param.axis;
    int32_t repeat = param.repeats;
    int32_t ndim = x->ndim;
    if(axis < 0) axis = axis + ndim;

#pragma omp parallel for
    for(uint64_t i = 0; i < getSize(y); i++){
      uint64_t o_i = i, in_i = 0, shapeSize = 1;
      for(int j = ndim-1; j >= 0; j--){
        uint64_t col = o_i % y->shape[j];
        o_i /= y->shape[j];
        if(j == axis) col = col / repeat;
        in_i += col * shapeSize;
        shapeSize *= x->shape[j];
      }
      y_data[i] = x_data[in_i];
    }
#ifdef CVM_PROFILING
    double end = omp_get_wtime();
    static double use_time = 0.0;
    use_time += end-start;
#endif
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

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.tile")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int32_t yndim = y->ndim;
    int32_t xndim = x->ndim;

    uint64_t tmp_y_size = 1;
    for(int i = 0; i < xndim; i++){
        tmp_y_size *= y->shape[i + yndim - xndim];
    }

    for(uint64_t i = 0; i < tmp_y_size; i++){
       uint64_t o_i = i, in_i = 0, shapeSize = 1;
       for(int j = xndim-1; j >= 0; j--){
            int yj = j + yndim - xndim;
            int col = o_i % y->shape[yj];
            o_i /= y->shape[yj];
            col = col % x->shape[j];
            in_i += col * shapeSize; 
            shapeSize *= x->shape[j];
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
    print_to_file(y, "tile.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.expand_dims")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
    DLTensor *ishape = args[0];
    DLTensor *oshape = args[1];
    int32_t *ishape_data = static_cast<int32_t*>(ishape->data);
    int32_t *oshape_data = static_cast<int32_t*>(oshape->data);
    if(ishape_data == oshape_data){
        return;
    }
    memcpy(oshape_data, ishape_data, getSize(ishape)* sizeof(int32_t));
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.squeeze")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
    DLTensor *ishape = args[0];
    DLTensor *oshape = args[1];
    int32_t *ishape_data = static_cast<int32_t*>(ishape->data);
    int32_t *oshape_data = static_cast<int32_t*>(oshape->data);
    if(ishape_data == oshape_data){
        return;
    }
    memcpy(oshape_data, ishape_data, getSize(ishape)* sizeof(int32_t));
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.transpose")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::TransposeParam>(attr->parsed);

    TShape axes = param.axes;
    for(uint32_t i = 0; i < axes.ndim(); i++){
        if(axes[i] < 0) axes[i] += x->ndim;
    }
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int ndim = y->ndim;
    if (axes.ndim() == 3 && axes[0] == 1 && axes[1] == 2 && axes[2] == 0) {
      int step = x->shape[1] * x->shape[2];
      for (int i = 0; i < step; i++) {
        for (int j = 0; j < x->shape[0]; j++) {
          y_data[i * x->shape[0]+ j ] = x_data[j * step + i];
        }
      }
    }
    else {
      for(uint64_t i = 0; i < getSize(y); i++) {
        uint64_t o_i = i, in_i = 0;
        for(int j = ndim - 1; j >= 0; j--){
          uint64_t col = o_i % y->shape[j];
          o_i /= y->shape[j];
          int xj = j;
          if(axes.ndim() > 0) {
            xj = axes[j];
          } else {
            xj = ndim - 1 - j;
          }
          int xi = 1;
          for(int tx = ndim-1; tx > xj; tx--){
            xi *= x->shape[tx];
          }
          in_i += col * xi;
        }
        y_data[i] = x_data[in_i];
      }
    }
    print_to_file(y, "transpose.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.strided_slice")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::StridedSliceParam>(attr->parsed);

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    TShape begin = param.begin;
    TShape end = param.end;
    TShape stride = param.stride;
    int ndim = y->ndim;
    int32_t num_axis = x->ndim;
    int64_t *dshp = x->shape;
    std::vector<int64_t> begin_vec;
    std::copy(begin.begin(), begin.end(), std::back_inserter(begin_vec));
    for (dim_t i = begin_vec.size(); i < num_axis; ++i) {
      begin_vec.push_back(0);
    }

    std::vector<int64_t> stride_vec;
    std::copy(stride.begin(), stride.end(), std::back_inserter(stride_vec));
    for (dim_t i = stride_vec.size(); i < num_axis; ++i) {
      stride_vec.push_back(1);
    }

    for (size_t i = 0; i < begin_vec.size(); ++i) {
      int64_t begin_range = stride_vec[i] < 0 ? -1 : 0;
      int64_t end_range = stride_vec[i] < 0 ? dshp[i] -1 : dshp[i];
      int64_t begin = begin_vec[i];
      if (begin < 0) begin += dshp[i];
      begin_vec[i]= std::min(std::max(begin, begin_range), end_range);
    }

    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t o_i = i, in_i = 0, shapeSize = 1;
        for(int j = ndim-1; j >= 0; j--){
            uint64_t col = o_i % y->shape[j];
            o_i /= y->shape[j];
            int64_t tbegin = begin_vec[j];
            int64_t tstep = stride_vec[j];
            col = tbegin + col * tstep;
            in_i += col * shapeSize;
            shapeSize *= x->shape[j];
        }
        y_data[i] = x_data[in_i];
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.slice_like")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[2];
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::SliceLikeParam>(attr->parsed);
    Tuple<int> axis = param.axis;

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int ndim = x->ndim;

    for(uint64_t i = 0; i < getSize(y); i++){
      uint64_t o_i = i, in_i = 0, shapeSize = 1;
      for(int j = ndim-1; j >= 0; j--){
        int col = o_i % y->shape[j];
        o_i /= y->shape[j];
        in_i += col * shapeSize;
        shapeSize *= x->shape[j];
      }
      y_data[i] = x_data[in_i];
    }
});

void take(DLTensor *x, DLTensor *indices, DLTensor *y){
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *indices_data = static_cast<int32_t*>(indices->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    uint64_t xs = getSize(x);

    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t in_i = std::min((uint64_t)std::max(indices_data[i], 0), xs-1);
        y_data[i] = x_data[in_i];
    }
}

void take(DLTensor *x, DLTensor *indices, DLTensor *y, const int32_t axis){
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *indices_data = static_cast<int32_t*>(indices->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int32_t yndim = y->ndim;
    int32_t xndim = x->ndim;
    int32_t indices_ndim = indices->ndim;
    if (axis == 0 && xndim == 2 && yndim == 3) {
      const int K = x->shape[1];
      uint64_t wn = getSize(indices);
      auto indices_data = static_cast<int32_t*>(indices->data);
      for (uint64_t row = 0; row < wn; row++) {
        uint64_t x_indices_i = std::min((int64_t)std::max(indices_data[row], 0), x->shape[0] - 1);
        memcpy(y_data +  row * K, x_data + x_indices_i * K, K * sizeof(int32_t));
      }
    }
    else {
      std::vector<size_t> x_shape_size(xndim, 1), indices_shape_size(indices_ndim, 1);
      for (int i = xndim-2; i >= 0; --i) {
        x_shape_size[i] = x_shape_size[i+1] * x->shape[i+1];
      }
      for (int i = indices_ndim-2; i >= 0; --i) {
        indices_shape_size[i] = indices_shape_size[i+1] * indices->shape[i+1];
      }
      for (size_t i = 0; i < getSize(y); ++i) {
        size_t oi = i, xi = 0, idxi = 0;
        for(int j = yndim - 1; j>=0; --j){
          size_t col = oi % y->shape[j];
          oi /= y->shape[j];
          if (axis <= j && j < axis+indices_ndim) {
            idxi += col * indices_shape_size[j - axis];
          } else {
            int xidx = j < axis ? j : j - indices_ndim + 1;
            xi += col * x_shape_size[xidx];
          }

          if (axis == j) {
            int64_t idxx = std::min(std::max(indices_data[idxi], 0), 
                (int32_t)x->shape[j]-1);
            xi += idxx * x_shape_size[j];
          }
        }
        y_data[i] = x_data[xi];
      }
    }
}

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.take")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *x = args[0];
    DLTensor *indices = args[1];
    DLTensor *y = args[2];
    void *_attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::TakeParam>(attr->parsed);

    if(!param.axis.has_value()){
      take(x, indices, y);
    }else{
      int32_t axis = param.axis.value();
      if(axis < 0){
          axis += x->ndim;
      }
      take(x, indices, y, axis);
    }
    print_to_file(y, "take.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.cvm_lut")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *indices = args[0];
    DLTensor *x = args[1];
    DLTensor *y = args[2];

    take(x, indices, y);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm.upsampling")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
#ifdef CVM_PROFILING
  double start = omp_get_wtime();
#endif
  DLTensor *x = args[0];
  DLTensor *y = args[1];

  void *_attr = args[2];
  auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
  auto &param = cvm::get<cvm::top::UpSamplingParam>(attr->parsed);

  uint32_t scale = {(uint32_t)param.scale};
  uint32_t h = x->shape[2], w = x->shape[3];
  uint32_t oh = y->shape[2], ow = y->shape[3];
  uint32_t n_batch = x->shape[0], n_channels = x->shape[1];

  auto x_data = static_cast<int32_t*>(x->data);
  auto y_data = static_cast<int32_t*>(y->data);

  //TODO(zkh) optimize nested for-loop for scale
  #pragma omp parallel for collapse(2)
  for (uint32_t batch = 0; batch < n_batch; ++batch) {
    for (uint32_t c = 0; c< n_channels; ++c) {
      auto bc_y_data = y_data + batch * n_channels * oh * ow + c * oh * ow;
      auto bc_x_data = x_data + batch * n_channels *  h *  w + c *  h *  w;
      for(uint32_t y = 0; y < oh; ++y){
        for(uint32_t x = 0; x < ow; ++x){
            bc_y_data[y * ow + x] = bc_x_data[y/scale * w + x/scale];
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



