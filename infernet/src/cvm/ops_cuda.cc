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
#include <immintrin.h>
#include "graph_runtime.h"

#ifdef CVM_RUNTIME_CUDA

#include "cuda_ops.h"

namespace cvm {
namespace runtime {
namespace cuda {

inline uint64_t getSize(DLTensor *dlTensor){
  uint64_t size = 1;
  for(int i = 0; i < dlTensor->ndim; i++){
    size *= dlTensor->shape[i];
  }
  return size;
}

void deal_error(int error_code, const char* errorStr){
    if(error_code == NON_ERROR){
        return;
    }
    if(error_code >= 10 && error_code < 20){
        VERIFY(false) << errorStr;
    }
    else{
        CHECK(false) << errorStr;
    }
}

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.elemwise_add")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 4);
    DLTensor *a = args[0];
    DLTensor *b = args[1];
    DLTensor *c = args[2];
    int32_t *a_data = static_cast<int32_t*>(a->data);
    int32_t *b_data = static_cast<int32_t*>(b->data);
    int32_t *c_data = static_cast<int32_t*>(c->data);
    uint64_t n = getSize(a);
    int error_code = NON_ERROR;
    const char *errorStr = cuda_elemwise_add(a_data, b_data, c_data, n, error_code);
    deal_error(error_code, errorStr);

});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.elemwise_sub")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 4);
    DLTensor *a = args[0];
    DLTensor *b = args[1];
    DLTensor *c = args[2];
    int32_t *a_data = static_cast<int32_t*>(a->data);
    int32_t *b_data = static_cast<int32_t*>(b->data);
    int32_t *c_data = static_cast<int32_t*>(c->data);
    uint64_t n = getSize(a);

    int error_code = NON_ERROR;
    const char *errorStr = cuda_elemwise_sub(a_data, b_data, c_data, n, error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.conv2d")
.set_body([](CVMArgs args, CVMRetValue* rv){
    VERIFY(args.num_args == 5 || args.num_args == 4);
    DLTensor *x = args[0];
    VERIFY(x->ndim == 4);
    DLTensor *w = args[1];
    VERIFY(w->ndim == 4);
    DLTensor *b = nullptr;
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
    int dilation[2] = {static_cast<int>(param.dilation[0]), static_cast<int>(param.dilation[1])};
    //int kernel_size[2] = {static_cast<int>(param.kernel_size[0]), static_cast<int>(param.kernel_size[1])};
    int padding[2] = {static_cast<int>(param.padding[0]), static_cast<int>(param.padding[1])};
    int strides[2] = {static_cast<int>(param.strides[0]), static_cast<int>(param.strides[1])};

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
    VERIFY(strides[0] > 0 && strides[1] > 0);
    int o_h = (x_h + 2 * padding[0] - t_filter_h) / strides[0] + 1;
    int o_w = (x_w + 2 * padding[1] - t_filter_w) / strides[1] + 1;
    if(n_batch < 1 || in_channels < 1 || x_h < 1 || x_w < 1 || filter_c < 1 || filter_h < 1 || filter_w < 1 ||
        padding[0] < 0 || padding[1] < 0 || stride_h < 1 || stride_w < 1 || dilation_h < 1 || dilation_w < 1 ||
        out_channels < 1 || o_h < 1 || o_w < 1){
      VERIFY(false) << "error args";
    }

    int error_code = NON_ERROR;
    const char* errorStr = "";
    if(groups == 1){
      errorStr = cuda_conv2d(
          x_data, n_batch, in_channels, x_h, x_w,
          w_data, out_channels, in_channels, filter_h, filter_w,
          b_data,
          padding[0], padding[1],
          strides[0], strides[1],
          dilation[0], dilation[1],
          groups,
          y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, error_code);
    }else{
      errorStr = cuda_depthwise_conv2d(
          x_data, n_batch, in_channels, x_h, x_w,
          w_data, out_channels, in_channels, filter_h, filter_w,
          b_data,
          padding[0], padding[1],
          strides[0], strides[1],
          dilation[0], dilation[1],
          groups,
          y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, error_code);
    }
    deal_error(error_code, errorStr);
 });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.dense")
.set_body([](CVMArgs args, CVMRetValue* rv) {
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
    int error_code = NON_ERROR;
    const char* errorStr = cuda_dense(
        dx, dw, dy,
        static_cast<int32_t>(x->shape[0]),
        static_cast<int32_t>(x->shape[1]),
        static_cast<int32_t>(y->shape[1]),
        db,
        error_code);

    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.clip")
.set_body([](CVMArgs args, CVMRetValue* rv) {
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto& param = cvm::get<cvm::top::ClipParam>(attr->parsed);
    int max = param.a_max;
    int min = param.a_min;

    int error_code = NON_ERROR;
    const char *errorStr = cuda_clip(
        static_cast<int32_t*>(x->data),
        static_cast<int32_t*>(y->data),
        getSize(x),
        max, min, error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.relu")
.set_body([](CVMArgs args, CVMRetValue* rv) {
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    int error_code = NON_ERROR;
    const char* errorStr = cuda_relu(
        static_cast<int32_t*>(x->data),
        static_cast<int32_t*>(y->data),
        getSize(x),
        error_code);
    deal_error(error_code, errorStr);
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.flatten")
.set_body([](CVMArgs args, CVMRetValue* rv){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];

    int error_code = NON_ERROR;
    const char* errorStr = cuda_flatten(
        static_cast<int32_t*>(x->data),
        static_cast<int32_t*>(y->data),
        getSize(x),
        error_code);
    deal_error(error_code, errorStr);
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_add")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 4);
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

    int error_code = NON_ERROR;
    const char* errorStr = cuda_broadcast_add(a, b, c, getSize(args0),
        ashape, adim,
        bshape, bdim,
        cshape, cdim, error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_sub")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 4);
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

    int error_code = NON_ERROR;
    const char* errorStr = cuda_broadcast_sub(a, b, c, getSize(args0),
        ashape, adim,
        bshape, bdim,
        cshape, cdim, error_code);
    deal_error(error_code, errorStr);
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_mul")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 4);
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

    int error_code = NON_ERROR;
    const char* errorStr = cuda_broadcast_mul(a, b, c, getSize(args0),
        ashape, adim,
        bshape, bdim,
        cshape, cdim, error_code);
    deal_error(error_code, errorStr);
});
//CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_div")
//    .set_body([](CVMArgs args, CVMRetValue *ret){
//        VERIFY(args.num_args == 4);
//        DLTensor *args0 = args[0];
//        DLTensor *args1 = args[1];
//        DLTensor *args2 = args[2];
//        int32_t *a = static_cast<int32_t*>(args0->data);
//        int32_t *b = static_cast<int32_t*>(args1->data);
//        int32_t *c = static_cast<int32_t*>(args2->data);
//        int64_t *ashape = static_cast<int64_t*>(args0->shape);
//        int32_t adim = static_cast<int32_t>(args0->ndim);
//        int64_t *bshape = static_cast<int64_t*>(args1->shape);
//        int32_t bdim = static_cast<int32_t>(args1->ndim);
//        int64_t *cshape = static_cast<int64_t*>(args2->shape);
//        int32_t cdim = static_cast<int32_t>(args2->ndim);
//
//
//        const char* errorStr = cuda_broadcast_div(a, b, c, getSize(args0),
//		ashape, adim,
//		bshape, bdim,
//		cshape, cdim, DEBUG_OP);
//
//        VERIFY_EQ(errorStr == NULL, true) << errorStr;
//    });
//CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_right_shift")
//    .set_body([](CVMArgs args, CVMRetValue *ret){
//        VERIFY(args.num_args == 4);
//        DLTensor *args0 = args[0];
//        DLTensor *args1 = args[1];
//        DLTensor *args2 = args[2];
//        int32_t *a = static_cast<int32_t*>(args0->data);
//        int32_t *b = static_cast<int32_t*>(args1->data);
//        int32_t *c = static_cast<int32_t*>(args2->data);
//        int64_t *ashape = static_cast<int64_t*>(args0->shape);
//        int32_t adim = static_cast<int32_t>(args0->ndim);
//        int64_t *bshape = static_cast<int64_t*>(args1->shape);
//        int32_t bdim = static_cast<int32_t>(args1->ndim);
//        int64_t *cshape = static_cast<int64_t*>(args2->shape);
//        int32_t cdim = static_cast<int32_t>(args2->ndim);
//
//        const char* errorStr = cuda_broadcast_right_shift(a, b, c, getSize(args0),
//		ashape, adim,
//		bshape, bdim,
//		cshape, cdim, DEBUG_OP);
//
//        VERIFY_EQ(errorStr == NULL, true) << errorStr;
//    });
//CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_left_shift")
//    .set_body([](CVMArgs args, CVMRetValue *ret){
//        VERIFY(args.num_args == 4);
//        DLTensor *args0 = args[0];
//        DLTensor *args1 = args[1];
//        DLTensor *args2 = args[2];
//        int32_t *a = static_cast<int32_t*>(args0->data);
//        int32_t *b = static_cast<int32_t*>(args1->data);
//        int32_t *c = static_cast<int32_t*>(args2->data);
//        int64_t *ashape = static_cast<int64_t*>(args0->shape);
//        int32_t adim = static_cast<int32_t>(args0->ndim);
//        int64_t *bshape = static_cast<int64_t*>(args1->shape);
//        int32_t bdim = static_cast<int32_t>(args1->ndim);
//        int64_t *cshape = static_cast<int64_t*>(args2->shape);
//        int32_t cdim = static_cast<int32_t>(args2->ndim);
//
//        const char* errorStr = cuda_broadcast_left_shift(a, b, c, getSize(args0),
//		ashape, adim,
//		bshape, bdim,
//		cshape, cdim, DEBUG_OP);
//
//        VERIFY_EQ(errorStr == NULL, true) << errorStr;
//    });

/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.max_pool2d")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::MaxPool2DParam>(attr->parsed);
    int strides[2] = {static_cast<int>(param.strides[0]), static_cast<int>(param.strides[1])};
    int pool_size[2] = {static_cast<int>(param.pool_size[0]), static_cast<int>(param.pool_size[1])};
    int padding[2] = {static_cast<int>(param.padding[0]), static_cast<int>(param.padding[0])};
    if(param.padding.ndim() == 2){
        padding[1] = static_cast<int>(param.padding[1]);
    }
   // bool ceil_mode = param.ceil_mode;

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

    int error_code = NON_ERROR;
    const char* errorStr = cuda_max_pool(
        x_data, n_batch, in_channels, x_h, x_w,
        filter_h, filter_w,
        padding[0], padding[1],
        stride_h, stride_w,
        y_data, n_batch, out_channels, o_h, o_w, x->ctx.device_id, error_code);
    deal_error(error_code, errorStr);
});

/*
* axis (2, 3)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.sum")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *dlx = args[0];
    DLTensor *y = args[1];
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t* x = static_cast<int32_t*>(dlx->data);
    void* _attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
    TShape axis = param.axis;
    int64_t *axis_data = axis.begin();
    for(size_t i = 0; i < axis.ndim(); i++){
    if(axis_data[i] < 0) axis_data[i] += dlx->ndim;
      VERIFY(axis_data[i] >= 0 && axis_data[i] < dlx->ndim);
    }
    //bool keepdims = param.keepdims;
    std::vector<int64_t> raxis;
    bool exclude = param.exclude;
    try{
      if(!exclude){
        for(size_t i = 0; i < axis.ndim(); i++){
          raxis.push_back(axis[i]);
        }
      }else{
        raxis.resize(dlx->ndim - axis.ndim());
        for(int32_t i = 0, k = 0; i < dlx->ndim; i++){
          bool flag = false;
          for(uint32_t j = 0; j < axis.ndim(); j++){
            if(axis_data[j] == i) {
              flag = true;
              break;
            }
          }
          if(!flag){
            raxis[k++] = i;
          }
        }
      }
    }catch(std::bad_alloc& e){
        CHECK(false) << e.what();
    }

    int error_code = NON_ERROR;
    const char* errorStr;
    if(exclude && raxis.size() == 0){
      errorStr = cuda_reshape(x, y_data, getSize(dlx), error_code);
      deal_error(error_code, errorStr);
    }
    else if(raxis.size() == 0){
      errorStr = cuda_sum(x, y_data, getSize(dlx), getSize(y),
          dlx->shape, y->shape, NULL, NULL,
          NULL, 0, dlx->ndim, y->ndim, raxis.size(), error_code);
      deal_error(error_code, errorStr);
    }else{
      try{
        std::vector<int32_t> realAxis(raxis.size());
        //std::vector<bool> flag(dlx->ndim, false);
        int32_t* flag = new int32_t[dlx->ndim];
        std::memset(flag, 0, sizeof(int32_t)*dlx->ndim);
        for(uint32_t i = 0; i < raxis.size(); i++){
          int32_t val = raxis[i];
          realAxis[i] = val;
          flag[val] = 1;
        }
        std::sort(realAxis.begin(), realAxis.end());
        realAxis.resize(std::unique(realAxis.begin(), realAxis.end()) - realAxis.begin());

        uint64_t axis_size = 1;
        for(uint32_t i = 0; i < realAxis.size(); i++){
          axis_size *= dlx->shape[realAxis[i]];
        }
        std::vector<uint64_t> every_xdim_size(dlx->ndim, 1);
        for(int i = dlx->ndim-2; i >= 0; i--){
          every_xdim_size[i] = dlx->shape[i+1] * every_xdim_size[i+1];
        }

        int32_t yndim = y->ndim;
        std::vector<int64_t> yshape(y->ndim);
        for(int32_t i = 0; i < y->ndim; i++){
          yshape[i] = y->shape[i];
        }
        for(int32_t i = 0, j = 0; i < y->ndim; i++){
          if(y->shape[i] == 1) {
            yndim -= 1;
          }else{
            yshape[j++] = y->shape[i];
          }
        }
        errorStr = cuda_sum(x, y_data, getSize(dlx), getSize(y),
            dlx->shape, yshape.data(), realAxis.data(), flag,
            every_xdim_size.data(), axis_size, dlx->ndim, yndim, raxis.size(), error_code);
        delete flag;
        deal_error(error_code, errorStr);
      }catch(const std::bad_alloc& e){
        CHECK(false) << e.what();
      }
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.reshape")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    //         std::string newshape = args[2];
    int error_code = NON_ERROR;
    const char* errorStr = cuda_reshape(
        static_cast<int32_t*>(x->data),
        static_cast<int32_t*>(y->data),
        getSize(x),
        error_code);
    deal_error(error_code, errorStr);
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
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::CVMClipParam>(attr->parsed);
    int32_t precision = param.precision;
    VERIFY(precision > 0) << "precision must greater zero";

    int error_code = NON_ERROR;
    const char* errorStr = cuda_cvm_clip(
        x_data,
        precision,
        y_data,
        getSize(x),
        error_code);
    deal_error(error_code, errorStr);
});

/*
 * a, input data
 * c, output data
 * precision, clip precision
 * b, shift b
 * */
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.cvm_right_shift")
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

    int error_code = NON_ERROR;
    const char* errorStr = cuda_cvm_right_shift(
        a_data,
        b,
        precision,
        c_data,
        getSize(a),
        error_code);
    deal_error(error_code, errorStr);
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.cvm_left_shift")
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
    int error_code = NON_ERROR;
    const char* errorStr = cuda_cvm_left_shift(
        a_data,
        b,
        precision,
        c_data,
        getSize(a),
        error_code);
    deal_error(error_code, errorStr);
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.log2")
.set_body([](CVMArgs args, CVMRetValue *ret){
    //        std::string x_str = args[0];
    VERIFY(args.num_args == 3);
    DLTensor *dlx = args[0];
    DLTensor *y = args[1];
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t *x = static_cast<int32_t*>(dlx->data);
    int error_code = NON_ERROR;
    const char* errorStr = cuda_log(x, y_data, error_code);
    deal_error(error_code, errorStr);
});
CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.abs")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *dlx = args[0];
    DLTensor *y = args[1];
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t* x = static_cast<int32_t*>(dlx->data);
    int error_code = NON_ERROR;
    const char* errorStr = cuda_abs(x, y_data, getSize(dlx), error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.max")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *dlx = args[0];
    DLTensor *y = args[1];
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t* x = static_cast<int32_t*>(dlx->data);
    void* _attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::ReduceParam>(attr->parsed);
    TShape axis = param.axis;
    int64_t *axis_data = axis.begin();
    for(size_t i = 0; i < axis.ndim(); i++){
    if(axis_data[i] < 0) axis_data[i] += dlx->ndim;
      VERIFY(axis_data[i] >= 0 && axis_data[i] < dlx->ndim);
    }
    //bool keepdims = param.keepdims;
    std::vector<int64_t> raxis;
    bool exclude = param.exclude;
    try{
      if(!exclude){
        for(size_t i = 0; i < axis.ndim(); i++){
          raxis.push_back(axis[i]);
        }
      }else{
        raxis.resize(dlx->ndim - axis.ndim());
        for(int i = 0, k = 0; i < dlx->ndim; i++){
          bool flag = false;
          for(size_t j = 0; j < axis.ndim(); j++){
            if(axis_data[j] == i) {
              flag = true;
              break;
            }
          }
          if(!flag){
            raxis[k++] = i;
          }
        }
      }
    }catch(const std::bad_alloc& e){
        CHECK(false) << e.what();
    }

    int error_code = NON_ERROR;
    const char* errorStr = "";
    if(exclude && raxis.size() == 0){
      errorStr = cuda_reshape(x, y_data, getSize(dlx), error_code);
      deal_error(error_code, errorStr);
    }
    else if(raxis.size() == 0){
      errorStr = cuda_max(x, y_data, getSize(dlx), getSize(y),
          dlx->shape, y->shape, NULL, NULL,
          NULL, 0, dlx->ndim, y->ndim, raxis.size(), error_code);
      deal_error(error_code, errorStr);
    }else{
      try{
        std::vector<int32_t> realAxis(raxis.size());
        //std::vector<bool> flag(dlx->ndim, false);
        int32_t* flag = new int32_t[dlx->ndim];
        std::memset(flag, 0, sizeof(int32_t)*dlx->ndim);
        for(uint32_t i = 0; i < raxis.size(); i++){
          int32_t val = raxis[i];
          realAxis[i] = val;
          flag[val] = 1;
        }
        std::sort(realAxis.begin(), realAxis.end());
        realAxis.resize(std::unique(realAxis.begin(), realAxis.end()) - realAxis.begin());

        uint64_t axis_size = 1;
        for(uint32_t i = 0; i < realAxis.size(); i++){
          axis_size *= dlx->shape[realAxis[i]];
        }
        std::vector<uint64_t> every_xdim_size(dlx->ndim, 1);
        for(int i = dlx->ndim-2; i >= 0; i--){
          every_xdim_size[i] = dlx->shape[i+1] * every_xdim_size[i+1];
        }

        int32_t yndim = y->ndim;
        std::vector<int64_t> yshape(y->ndim);
        for(int i = 0; i < y->ndim; i++){
          yshape[i] = y->shape[i];
        }
        for(int i = 0, j = 0; i < y->ndim; i++){
          if(y->shape[i] == 1) {
            yndim -= 1;
          }else{
            yshape[j++] = y->shape[i];
          }
        }
        errorStr = cuda_max(x, y_data, getSize(dlx), getSize(y),
            dlx->shape, yshape.data(), realAxis.data(), flag,
            every_xdim_size.data(), axis_size, dlx->ndim, yndim, raxis.size(), error_code);
        delete flag;
        deal_error(error_code, errorStr);
      }catch(const std::bad_alloc& e){
        CHECK(false) << e.what();
      }
    }
});

//CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.broadcast_max")
//    .set_body([](CVMArgs args, CVMRetValue *ret){
//        VERIFY(args.num_args == 4);
//        DLTensor *a = args[0];
//        DLTensor *b = args[1];
//        DLTensor *c = args[2];
//        int32_t *a_data = static_cast<int32_t*>(a->data);
//        int32_t* b_data = static_cast<int32_t*>(b->data);
//        int32_t* c_data = static_cast<int32_t*>(c->data);
//        int64_t *ashape = static_cast<int64_t*>(a->shape);
//        int32_t adim = static_cast<int32_t>(a->ndim);
//        int64_t *bshape = static_cast<int64_t*>(b->shape);
//        int32_t bdim = static_cast<int32_t>(b->ndim);
//        int64_t *cshape = static_cast<int64_t*>(c->shape);
//        int32_t cdim = static_cast<int32_t>(c->ndim);
//
//        const char* errorStr = cuda_broadcast_max(a_data, b_data, c_data, getSize(a),
//		ashape, adim,
//		bshape, bdim,
//		cshape, cdim, DEBUG_OP);
//        VERIFY(errorStr == NULL) << errorStr;
//    });

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.concatenate")
.set_body([](CVMArgs args, CVMRetValue *ret){
    int len = args.num_args;
    VERIFY(len >= 3);
    DLTensor *input0 = args[0];
    void *_attr = args[--len];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::ConcatenateParam>(attr->parsed);
    DLTensor *output = args[--len];
    int32_t axis = param.axis;
    int32_t ndim = static_cast<int32_t>(input0->ndim);
    VERIFY(-ndim <= axis && axis < ndim);
    if(axis < 0) axis += ndim;
    VERIFY(axis < input0->ndim && axis >= 0);

    int32_t *out_data = static_cast<int32_t*>(output->data);
    int64_t preSize = 0;
    for(int i = 0; i < len; i++){
      DLTensor *input  = args[i];
      int error_code = NON_ERROR;
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
          error_code
        );

      deal_error(error_code, errorStr);
      preSize += input->shape[axis];
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.repeat")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::RepeatParam>(attr->parsed);
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t axis = param.axis;
    int32_t repeat = param.repeats;
    int ndim = x->ndim;
    if(axis < 0) axis = axis + ndim;
    VERIFY(axis >= 0 && axis < ndim);

    int error_code = NON_ERROR;
    const char* errorStr = cuda_repeat(
            x_data, y_data, x->shape, y->shape, getSize(y), x->ndim, y->ndim, axis, repeat, error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.negative")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    VERIFY(x->ndim == y->ndim);

    int error_code = NON_ERROR;
    const char* errorStr = cuda_negative(x_data, y_data, getSize(y), error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.tile")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void* _attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::TileParam>(attr->parsed);

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int32_t yndim = y->ndim;
    int32_t xndim = x->ndim;
    TShape ts_reps = param.reps;
    int64_t *reps = ts_reps.begin();
    for(uint32_t i = 0; i < ts_reps.ndim(); i++){
        VERIFY(reps[i] > 0);
    }

    int i = 0, j = 0, k = 0;
    for(i = yndim-1, j = xndim-1, k = ts_reps.ndim()-1; i >= 0 && j >= 0 && k >= 0; i--, j--, k--){
        VERIFY(x->shape[j] * reps[k] == y->shape[i]);
    }
    for(; i >= 0 && k >= 0; i--, k--){
        VERIFY(reps[k] == y->shape[i]);
    }

    int error_code = NON_ERROR;
    const char* errorStr = cuda_tile(x_data, y_data, getSize(y), yndim, xndim, x->shape, y->shape, error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.expand_dims")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
    DLTensor *ishape = args[0];
    DLTensor *oshape = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::ExpandDimsParam>(attr->parsed);

    int32_t axis = param.axis;
   // axis = axis < 0 ? axis + ishape->ndim : axis;
   // VERIFY(axis >= 0 && axis <= ishape->ndim);
    int32_t *ishape_data = static_cast<int32_t*>(ishape->data);
    int32_t *oshape_data = static_cast<int32_t*>(oshape->data);

    int error_code = NON_ERROR;
    const char* errorStr = cuda_expand_dims(ishape_data, oshape_data, axis, getSize(oshape), error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.transpose")
.set_body([](CVMArgs args, CVMRetValue *ret){
    int num_args = args.num_args;
    VERIFY(num_args == 3);
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::TransposeParam>(attr->parsed);

    TShape axes = param.axes;
    int64_t *axes_data = axes.begin();
    VERIFY(axes.ndim() == 0 || axes.ndim() == (uint32_t)x->ndim);
    for(uint32_t i = 0; i < axes.ndim(); i++){
        if(axes_data[i] < 0) axes_data[i] += x->ndim;
        VERIFY(axes_data[i] >= 0 && axes_data[i] < x->ndim);
    }

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int ndim = y->ndim;
    int error_code = NON_ERROR;
    const char* errorStr = cuda_transpose(x_data, axes_data, y_data, x->shape, y->shape, ndim,
            getSize(y), axes.ndim(), error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.strided_slice")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 3);
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

    int32_t num_axis = x->ndim;
    std::vector<int64_t> begin_vec;
    std::vector<int64_t> end_vec;
    std::vector<int64_t> stride_vec;

    try{
      std::copy(begin.begin(), begin.end(), std::back_inserter(begin_vec));
      for (dim_t i = begin_vec.size(); i < num_axis; ++i) {
        begin_vec.push_back(0);
      }

      std::copy(end.begin(), end.end(), std::back_inserter(end_vec));
      for (dim_t i = end_vec.size(); i < num_axis; ++i) {
        end_vec.push_back(x->shape[i]);
      }

      std::copy(stride.begin(), stride.end(), std::back_inserter(stride_vec));
      for (dim_t i = stride_vec.size(); i < num_axis; ++i) {
        stride_vec.push_back(1);
      }
    }catch(const std::bad_alloc& e){
        CHECK(false) << e.what();
    }

    int64_t *begin_data = begin_vec.data();//begin.begin();
    int64_t *end_data = end_vec.data();//end.begin();
    int64_t *step_data = stride_vec.data();//stride.begin();

    for(int32_t i = 0; i < num_axis; i++){
        if(begin_data[i] < 0) {
          begin_data[i] += x->shape[i];
          begin_data[i] = std::min(std::max(begin_data[i], (int64_t)0), (int64_t)x->shape[i]-1);
        }
       // if(end_data[i] < 0) {
       //   end_data[i] += x->shape[i];
       //   end_data[i] += std::min(std::max(end_data[i], (int64_t)0), (int64_t)x->shape[i]-1);
       // }
       // VERIFY(step_data[i] != 0);
       // if(step_data[i] > 0) {
       //   VERIFY(begin_data[i] < end_data[i]);
       // }else{
       //   VERIFY(begin_data[i] > end_data[i]);
       // }
    }

    int error_code = NON_ERROR;
    const char *errorStr = cuda_stride_slice(x_data, y_data, begin_data, begin.ndim(), step_data,
            x->shape, y->shape, stride.ndim(), y->ndim, getSize(y), x->ndim, error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.slice_like")
.set_body([](CVMArgs args, CVMRetValue *ret){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    //DLTensor *shape = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::SliceLikeParam>(attr->parsed);
    Tuple<int> axis = param.axis;
   // int *axis_data = axis.begin();

    int32_t *x_data = static_cast<int32_t*>(x->data);
    //  int32_t *shape_like = static_cast<int32_t*>(shape->data);
    //VERIFY(axis.ndim() < (uint32_t)x->ndim && axis.ndim() <= (uint32_t)shape->ndim);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int ndim = x->ndim;

    int error_code = NON_ERROR;
    const char *errorStr = cuda_slice_like(x_data, y_data, x->shape, y->shape, getSize(y), ndim, error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.get_valid_counts")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    DLTensor *valid_count = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::GetValidCountsParam>(attr->parsed);

    int32_t score_threshold = param.score_threshold;

    VERIFY(x->ndim == 3);
    int32_t batchs = x->shape[0];
    int32_t n = x->shape[1];
    int32_t k = x->shape[2];
    VERIFY(k >= 2);

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *valid_count_data = static_cast<int32_t*>(valid_count->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int error_code = NON_ERROR;
    const char* errorStr = cuda_get_valid_counts(x_data, y_data, valid_count_data, n, k, score_threshold, batchs, error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.non_max_suppression")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    DLTensor *valid_count = args[1];
    DLTensor *y = args[2];
    void* _attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::NonMaximumSuppressionParam>(attr->parsed);

    int32_t max_output_size = param.max_output_size;
    int32_t iou_threshold = param.iou_threshold;
    int32_t topk = param.top_k;
    int32_t coord_start = param.coord_start;
    int32_t score_index = param.score_index;
    int32_t id_index = param.id_index;
    bool force_suppress = param.force_suppress;
    bool return_indices = param.return_indices;
    //bool invalid_to_bottom = param.invalid_to_bottom;
    CHECK(return_indices == false) << "no support return_indices and invalid_to_bottom";

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *valid_count_data = static_cast<int32_t*>(valid_count->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    VERIFY(x->ndim == 3);
    int32_t batchs = x->shape[0];
    int32_t n = x->shape[1];
    int32_t k = x->shape[2];
    VERIFY(k == 6);

    int error_code = NON_ERROR;
    const char* errorStr = cuda_non_max_suppression(
            x_data, valid_count_data, y_data, batchs, n, k,
            max_output_size, iou_threshold, topk, coord_start, score_index, id_index, force_suppress,
            error_code);
    deal_error(error_code, errorStr);
});

//CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.bias_add")
//.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
//    DLTensor *x = args[0];
//    DLTensor *bias = args[1];
//    DLTensor *y = args[2];
//    int32_t axis = 1;
//    int32_t ndim = x->ndim;
//    VERIFY(axis > 0 && axis < ndim);
//
//    const int32_t *x_data = static_cast<int32_t*>(x->data);
//    const int32_t *bias_data = static_cast<int32_t*>(bias->data);
//    int32_t *y_data = static_cast<int32_t*>(y->data);
//    const char* errorStr = cuda_bias_add(x_data, bias_data, y_data, getSize(y), y->shape, ndim, axis);
//    VERIFY_EQ(errorStr == NULL, true) << errorStr;
//});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.upsampling")
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

    uint32_t scale = {(uint32_t)param.scale};
    uint32_t h = x->shape[2], w = x->shape[3];
    uint32_t oh = y->shape[2], ow = y->shape[3];
    uint32_t n_batch = x->shape[0], n_channels = x->shape[1];

    auto x_data = static_cast<int32_t*>(x->data);
    auto y_data = static_cast<int32_t*>(y->data);

    int error_code = NON_ERROR;
    const char* errorStr = cuda_upsampling_nearest(x_data, y_data, scale, h, w, oh, ow, n_batch, n_channels, error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.take")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    DLTensor *indices = args[1];
    DLTensor *y = args[2];
    void *_attr = args[3];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::TakeParam>(attr->parsed);

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *indices_data = static_cast<int32_t*>(indices->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    // take(x, indices, y, axis);
    // std::cerr << "cuda take axis = " << axis << " ysize = " << getSize(y) <<  "\n";
    int error_code = NON_ERROR;
    const char *errorStr = "";

    if(param.axis.has_value()){
      int32_t axis = param.axis.value();
      if(axis < 0){
          axis += x->ndim;
      }
      errorStr = cuda_take(x_data, indices_data, y_data, x->shape, y->shape,
              indices->shape, y->ndim, x->ndim, indices->ndim, getSize(y), axis, error_code);
    }else{
      errorStr = cuda_take(x_data, indices_data, y_data, getSize(y), getSize(x), error_code);
    }
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.cvm_lut")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    VERIFY(args.num_args == 4);
    DLTensor *x = args[0];
    DLTensor *indices = args[1];
    DLTensor *y = args[2];

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *indices_data = static_cast<int32_t*>(indices->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
//    take(indices, x, y);
    int error_code = NON_ERROR;
    const char* errorStr = cuda_take(indices_data, x_data, y_data, getSize(y), getSize(x), error_code);
    deal_error(error_code, errorStr);
});

CVM_REGISTER_GLOBAL("cvm.runtime.cvm_cuda.squeeze")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
    VERIFY(args.num_args == 3);
    DLTensor *ishape = args[0];
    DLTensor *oshape = args[1];
    // void *_attr = args[2];
    // auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    // auto &param = cvm::get<cvm::top::SqueezeParam>(attr->parsed);
    int32_t *ishape_data = static_cast<int32_t*>(ishape->data);
    int32_t *oshape_data = static_cast<int32_t*>(oshape->data);
    if(ishape_data == oshape_data){
        return;
    }
    int error_code = NON_ERROR;
    const char* errorStr = cuda_squeeze(ishape_data, oshape_data, getSize(ishape), error_code);
    deal_error(error_code, errorStr);
});

}
}
}

#endif // end of CVM_RUNTIME_CUDA
