#include "ops.h"

namespace cvm {
namespace runtime {

CVM_REGISTER_GLOBAL("cvm.runtime.formal.relu")
.set_body([](CVMArgs args, CVMRetValue* rv){
   DLTensor *x = args[0];
   DLTensor *y = args[1];
   int32_t *x_data = static_cast<int32_t*>(x->data);
   int32_t *y_data = static_cast<int32_t*>(y->data);
   for (uint64_t i = 0; i < getSize(x); i++) {
        auto tmp = x_data[i];
        if (tmp < 0) tmp = 0;
        y_data[i] = tmp;
   }
  print_to_file(y, "relu.txt");
});

/*
* x : M*K
* w : N*K
* b : N
* y : M*N
/data/std_out/shufflenet*/
CVM_REGISTER_GLOBAL("cvm.runtime.formal.dense")
.set_body([](CVMArgs args, CVMRetValue* rv) {
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
    for (int64_t di = 0; di < y->shape[0]; ++di) {
      int32_t y_offset = di * y->shape[1];
      for (int64_t oi = 0; oi < y->shape[1]; ++oi) {
        y_data[y_offset + oi] += bias_data[oi];
      }
    }
  }
  print_to_file(y, "dense.txt");

});

void conv2d(
    int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
    int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
    int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
    int32_t *b_data,
    int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w){
  for(int32_t n = 0; n < n_batch; n++){
    for(int32_t oc = 0; oc < out_channels; oc++){
      for(int32_t oh = 0; oh < o_h; oh++){
        for(int32_t ow = 0; ow < o_w; ow++){
          int32_t sum = 0;
          for(int32_t ic = 0; ic < in_channels; ic++){
            for(int32_t fh = 0; fh < filter_h; fh++){
              for(int32_t fw = 0; fw < filter_w; fw++){
                int32_t ih = oh * stride_h + fh * dilation_h- padding[0];
                int32_t iw = ow * stride_w + fw * dilation_w- padding[1];
                if(ih < 0 || ih >= x_h || iw < 0 || iw >= x_w){
                  continue;
                }
                int32_t w_index = oc * filter_c * filter_h * filter_w + ic * filter_h * filter_w + fh * filter_w + fw;
                int32_t x_index = n * in_channels * x_h * x_w + ic * x_h * x_w + ih * x_w + iw;
                sum += w_data[w_index] * x_data[x_index];
              }
            }
          }
          int32_t y_index = n * out_channels * o_h * o_w + oc * o_h * o_w + oh * o_w + ow;
          y_data[y_index] = sum + (b_data != nullptr ? b_data[oc] : 0);
        }
      }
    }
  }
}

static void groupwise_conv2d(
   int32_t *x_data, int32_t n_batch, int32_t in_channels, int32_t x_h, int32_t x_w,
   int32_t *w_data, int32_t filter_c, int32_t filter_h, int32_t filter_w,
   int32_t *y_data, int32_t out_channels, int32_t o_h, int32_t o_w,
   int32_t *b_data,
   int32_t padding[2], int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
   int32_t groups){
  int32_t ochannels_per_group = out_channels / groups;
  int32_t ichannels_per_group = in_channels / groups;
  for(int32_t n = 0; n < n_batch; ++n){
    for(int32_t oc = 0; oc < out_channels; ++oc){
      for(int32_t oh = 0; oh < o_h; ++oh){
        for(int32_t ow = 0; ow < o_w; ++ow){
          int32_t oi = n * out_channels * o_h * o_w + oc * o_h * o_w + oh * o_w + ow;
          int32_t sum = 0;
          int32_t ic = oc / ochannels_per_group * ichannels_per_group;
          for(int32_t tic = 0; tic < ichannels_per_group; ++tic){
            for(int32_t fh = 0; fh < filter_h; ++fh){
              for(int32_t fw = 0; fw < filter_w; ++fw){
                int32_t th = oh * stride_h + fh*dilation_h - padding[0];
                int32_t tw = ow * stride_w + fw*dilation_w - padding[1];
                if(th < 0 || tw < 0 || th >= x_h || tw >= x_w)
                  continue;
                sum += x_data[n * in_channels * x_h * x_w + (ic+tic) * x_h * x_w + th * x_w + tw]
                  * w_data[oc * filter_c * filter_h * filter_w + tic * filter_h * filter_w + fh * filter_w + fw];
              }
            }
          }
          y_data[oi] = sum + (b_data == nullptr ? 0 : b_data[oc]);
        }
      }
    }
  }
}

CVM_REGISTER_GLOBAL("cvm.runtime.formal.conv2d")
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
  //int dilation_h = dilation[0];
  //int dilation_w = dilation[1];

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
    groupwise_conv2d(
        x_data, n_batch, in_channels, x_h, x_w,
        w_data, filter_c, filter_h, filter_w,
        y_data, out_channels, o_h, o_w,
        b_data,
        padding, stride_h, stride_w, dilation[0], dilation[1],
        groups);
  } else {
    conv2d(
        x_data, n_batch, in_channels, x_h, x_w,
        w_data, filter_c, filter_h, filter_w,
        y_data, out_channels, o_h, o_w,
        b_data,
        padding, stride_h, stride_w, dilation[0], dilation[1]);
  }
  print_to_file(y, "conv2d.txt");
});



/*
* strides (2, 2)
* pool_size [3, 3]
* ceil_mode False
* padding (1, 1)
*/
CVM_REGISTER_GLOBAL("cvm.runtime.formal.max_pool2d")
    .set_body([](CVMArgs args, CVMRetValue *ret)
{
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
    const int32_t minV = int32_t(1) << 31;
    int32_t y_max = minV;
    for (int r = 0; r < filter_h; ++r) {
      for (int s = 0; s < filter_w; ++s) {
        int32_t tp = p * stride_h + r - padding[0];
        int32_t tq = q * stride_w + s - padding[1];
        int32_t x_tmp = minV; 
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
  print_to_file(y, "max_pool.txt");

});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.cvm_precision")
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

CVM_REGISTER_GLOBAL("cvm.runtime.formal.abs")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t* x_data = static_cast<int32_t*>(x->data);
    for(uint64_t i = 0; i < getSize(x); i++){
      y_data[i] = std::abs(x_data[i]);
    }
});

// CVM_REGISTER_GLOBAL("cvm.runtime.formal.sqrt")
// .set_body([](CVMArgs args, CVMRetValue *ret){
    // DLTensor *x = args[0];
    // DLTensor *y = args[1];
    // int32_t *y_data = static_cast<int32_t*>(y->data);
    // int32_t* x_data = static_cast<int32_t*>(x->data);
    // for(uint64_t i = 0; i < getSize(x); i++){
      // y_data[i] = x_data[i] < 0 ? 0 : static_cast<int32_t>(std::sqrt(x_data[i]));
    // }
// });

CVM_REGISTER_GLOBAL("cvm.runtime.formal.concatenate")
.set_body([](CVMArgs args, CVMRetValue *ret){
    int M = args.num_args - 2; // I^0, I^1, ... I^M-1
    auto Y = CVMArg2Data<int32_t>(args[M]);
    auto params = CVMArg2Attr<top::ConcatenateParam>(args[M+1]);

    auto y_shape = CVMArgShape(args[M]);

    int32_t axis = params.axis;
    if(axis < 0) axis += y_shape.size();

    int64_t y_size = 1;
    for (int i = 0; i < axis; ++i) y_size *= y_shape[i];
    int32_t axis_batch = 1;
    for (size_t i = axis+1; i < y_shape.size(); ++i) axis_batch *= y_shape[i];

    int64_t y_start_idx = 0;
    int64_t y_axis_batch = y_shape.at(axis) * axis_batch;
    for (int m = 0; m < M; ++m) {
      auto Ix = CVMArg2Data<int32_t>(args[m]);
      auto x_shape = CVMArgShape(args[m]);
      auto x_axis_batch = x_shape.at(axis) * axis_batch;

      for (int64_t y_iter = 0; y_iter < y_size; ++y_iter) {
        memcpy(Y+y_iter*y_axis_batch+y_start_idx,
               Ix+y_iter*x_axis_batch,
               x_axis_batch*sizeof(int32_t));
      }

      y_start_idx += x_axis_batch;
    }

});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.repeat")
.set_body([](CVMArgs args, CVMRetValue *ret){
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
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.negative")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    for(uint64_t i = 0; i < getSize(x); i++){
        y_data[i] = -x_data[i];
    }
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.tile")
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

// CVM_REGISTER_GLOBAL("cvm.runtime.formal.pad")
// .set_body([](CVMArgs args, CVMRetValue *ret){
    // DLTensor *x = args[0];
    // DLTensor *y = args[1];
    // void *_attr = args[2];
    // auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    // auto &param = cvm::get<cvm::top::PadParam>(attr->parsed);

    // TShape pad_width = param.pad_width;
    // int pad_value = param.pad_value;
    // int32_t *x_data = static_cast<int32_t*>(x->data);
    // int32_t *y_data = static_cast<int32_t*>(y->data);

    // int32_t yndim = y->ndim;
    // for (uint64_t i = 0; i < getSize(y); i++) {
      // uint64_t o_i = i, in_i = 0, shapeSize = 1;
      // bool flag = true;
      // for (int j = xndim-1; j >= 0; j--) {
        // int col = o_i % y->shape[j];
        // int lower = pad_width[2*j], upper = x->shape[j]+pad_width[2*j];
        // if (col < lower || col >= upper) {
          // flag = false;
          // break;
        // }
        // o_i /= y->shape[j];
        // in_i += (col-lower) * shapeSize;
        // shapeSize *= x->shape[j];
      // }
      // y_data[i] = flag ? x_data[in_i] : pad_value;
    // }

    // print_to_file(y, "tile.txt");
// });

CVM_REGISTER_GLOBAL("cvm.runtime.formal.expand_dims")
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

CVM_REGISTER_GLOBAL("cvm.runtime.formal.squeeze")
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

CVM_REGISTER_GLOBAL("cvm.runtime.formal.transpose")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *x = args[0];
    DLTensor *y = args[1];
    void *_attr = args[2];
    auto *attr = static_cast<cvm::NodeAttrs*>(_attr);
    auto &param = cvm::get<cvm::top::TransposeParam>(attr->parsed);

    int32_t ndim = y->ndim;
    //TShape axes = param.axes;
    std::vector<int32_t> axes(ndim);
    for(int32_t i = 0; i < ndim; i++){
        if(param.axes.ndim() == 0){
          axes[i] = ndim - 1 - i;
        }else{
          int32_t axis = param.axes[i];
          axes[i] = axis < 0 ? axis + ndim : axis;
        }
    }
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    for(uint64_t i = 0; i < getSize(y); i++) {
      uint64_t o_i = i, in_i = 0;
      for(int j = ndim - 1; j >= 0; j--){
        uint64_t col = o_i % y->shape[j];
        o_i /= y->shape[j];
        int xi = 1;
        for(int tx = ndim-1; tx > axes[j]; tx--){
          xi *= x->shape[tx];
        }
        in_i += col * xi;
      }
      y_data[i] = x_data[in_i];
    }
    print_to_file(y, "transpose.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.strided_slice")
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
    print_to_file(y, "stride_slice.txt");
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.slice_like")
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

static void take(DLTensor *x, DLTensor *indices, DLTensor *y){
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *indices_data = static_cast<int32_t*>(indices->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    uint64_t xs = getSize(x);

    for(uint64_t i = 0; i < getSize(y); i++){
        uint64_t in_i = std::min((uint64_t)std::max(indices_data[i], 0), xs-1);
        y_data[i] = x_data[in_i];
    }
}

static void take(DLTensor *x, 
                 DLTensor *indices, 
                 DLTensor *y, 
                 const int32_t axis){
    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *indices_data = static_cast<int32_t*>(indices->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);

    int32_t yndim = y->ndim;
    int32_t xndim = x->ndim;
    int32_t indices_ndim = indices->ndim;
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

CVM_REGISTER_GLOBAL("cvm.runtime.formal.take")
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

CVM_REGISTER_GLOBAL("cvm.runtime.formal.cvm_lut")
.set_body([](cvm::runtime::CVMArgs args, cvm::runtime::CVMRetValue *rv){
    DLTensor *indices = args[0];
    DLTensor *x = args[1];
    DLTensor *y = args[2];

    take(x, indices, y);
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.upsampling")
.set_body([](CVMArgs args, CVMRetValue *ret)
{
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
});

CVM_REGISTER_GLOBAL("cvm.runtime.formal.where")
.set_body([](CVMArgs args, CVMRetValue *ret){
    DLTensor *condition = args[0];
    DLTensor *x = args[1];
    DLTensor *y = args[2];
    DLTensor *result = args[3];

    int32_t *x_data = static_cast<int32_t*>(x->data);
    int32_t *y_data = static_cast<int32_t*>(y->data);
    int32_t *condition_data = static_cast<int32_t*>(condition->data);
    int32_t *result_data = static_cast<int32_t*>(result->data);

    if(x->ndim == condition->ndim){
      for(uint64_t i = 0; i < getSize(result); ++i){
        result_data[i] = condition_data[i] == 0 ? y_data[i] : x_data[i];
      }
    }else{
      uint64_t size = 1;
      for(int32_t i = 1; i < result->ndim; i++){
        size *= result->shape[i];
      }
      for(int32_t i = 0; i < result->shape[0]; ++i){
        memcpy(&result_data[i*size], (condition_data[i] == 0 ? &y_data[i*size] : &x_data[i*size]), size); 
      } 
    } 
    print_to_file(result, "where.txt");
});

}
}



