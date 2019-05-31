#ifndef CUDA_OP_H
#define CUDA_OP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const char* cuda_elemwise_add(int32_t *a, int32_t *b, int32_t *c, int32_t n, bool debug);
const char* cuda_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t stride_w,
        int32_t dilation_h, int32_t dilation_w,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, bool debug);
const char* cuda_depthwise_conv2d(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t *filter, int32_t f_n, int32_t f_c, int32_t f_h, int32_t f_w,
        int32_t *bias,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t stride_w,
        int32_t dilation_h, int32_t dilation_w,
        int32_t groups,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, bool debug);
const char* cuda_max_pool(
        int32_t *input, int32_t i_n, int32_t i_c, int32_t i_h, int32_t i_w,
        int32_t f_h, int32_t f_w,
        int32_t padding_h, int32_t padding_w,
        int32_t stride_h, int32_t w,
        int32_t *output, int32_t o_n, int32_t o_c, int32_t o_h, int32_t o_w, int32_t device_id, bool debug);
const char* cuda_dense(
        int32_t *a,
        int32_t *b,
        int32_t *c,
        const int m, const int k, const int n, int32_t *bias, bool debug);
const char* cuda_clip(const int32_t *x, int32_t *y, const int32_t n, const int32_t max, const int32_t min, bool debug);
const char* cuda_relu(const int32_t *x, int32_t *y, const int32_t n, bool debug);
const char* cuda_flatten(const int32_t *x, int32_t *y, const int32_t n, bool debug);
const char* cuda_broadcast_add(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, bool debug);
const char* cuda_broadcast_sub(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, bool debug);
const char* cuda_broadcast_mul(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, bool debug);
const char* cuda_broadcast_div(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, bool debug);
const char* cuda_broadcast_right_shift(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, bool debug);
const char* cuda_broadcast_left_shift(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, bool debug);
const char* cuda_broadcast_max(const int32_t *a, const int32_t *b, int32_t* c, const int32_t n,
        int64_t* ashape, int32_t adim,
        int64_t* bshape, int32_t bdim,
        int64_t* cshape, int32_t cdim, bool debug);
const char* cuda_sum(
        const int32_t *x,
        const int32_t n_batch, const int32_t channels, const int32_t h, const int32_t w,
        int32_t *y, bool debug);
const char* cuda_reshape(const int32_t *x, int32_t *y, int32_t size, bool debug);
const char* cuda_log(const int32_t *x, int32_t *y, bool debug);
const char* cuda_abs(const int32_t *x, int32_t *y, const int32_t n, bool debug);
const char* cuda_max(const int32_t *x, int32_t *y, const int32_t n, bool debug);
const char* cuda_cvm_clip(const int32_t* x, const int32_t precision, int32_t *y, const int32_t n, bool debug);
const char* cuda_cvm_right_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const int32_t n, bool debug);
const char* cuda_cvm_left_shift(const int32_t *a, const int32_t b, const int32_t precision, int32_t *c, const int32_t n, bool debug);
const char* cuda_concatenate(const int32_t *input, const int64_t *ishape, const int32_t idim, const int32_t in,
        int32_t *output, int64_t* oshape, const int32_t odim, const int32_t on,
        const int64_t preShapeSize, const int64_t curShapeSize, const int32_t axis, bool debug);
const char* cuda_take(const int32_t *x_data, const int32_t *indices_data, int32_t *y_data, const int32_t axis, const int64_t ysize,
        const int64_t *xshape, const int64_t *indices_shape, const int64_t *yshape, const int32_t yndim, const int32_t xndim, const int32_t indices_ndim);
#endif
