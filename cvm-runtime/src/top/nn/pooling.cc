
/*!
 *  Copyright (c) 2017 by Contributors
 * \file pooling.cc
 * \brief Property def of pooling operators.
 */
#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/op_attr_types.h>
#include <cvm/top/nn.h>
#include "nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {

CVMUTIL_REGISTER_PARAMETER(MaxPool2DParam);

template <typename T>
inline bool Pool2DInferShape(const cvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_shape,
                             std::vector<TShape>* out_shape) {
  const T& param = cvm::get<T>(attrs.parsed);
  VERIFY_EQ(in_shape->size(), 1U);
  VERIFY_EQ(out_shape->size(), 1U);

  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() ==  0) return false;

  VERIFY_EQ(dshape.ndim(), 4U)
    << "Pool2D only support input = 4-D: NCHW";
  // VERIFY_GE(dshape.ndim(), 2U)
  //   << "Pool2D only support input >= 2-D: input must have height and width";

  VERIFY_EQ(param.layout, "NCHW")
    << "Pool2D only supported NCHW layout vs. " << param.layout;
  Layout layout(param.layout);
  // VERIFY(layout.contains('H') && layout.contains('W') &&
  //       !layout.contains('h') && !layout.contains('w'))
  //   << "Invalid layout " << layout
  //   << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.indexof('H');
  const auto widx = layout.indexof('W');

  dim_t pad_h, pad_w;
  VERIFY(param.padding.ndim() == 1U || param.padding.ndim() == 2U)
    << "Pool2D only supported 1-D or 2-D padding vs. "
    << param.padding;
  VerifyAttrRange(param.padding[0], "MaxPool2D.padding[0]");
  if (param.padding.ndim() == 1) {
    pad_h = param.padding[0] * 2;
    pad_w = param.padding[0] * 2;
  } else if (param.padding.ndim() == 2) {
    // (top, left)
    pad_h = param.padding[0] * 2;
    pad_w = param.padding[1] * 2;
    VerifyAttrRange(param.padding[1], "MaxPool2D.padding[1]");
  } else {
    return false;
  }

  TShape oshape = dshape;
  VERIFY(param.pool_size.ndim() == 2);
  VERIFY(param.strides.ndim() == 2);
  VerifyAttrRange(param.strides[0], "MaxPool2D.strides[0]", 1);
  VerifyAttrRange(param.strides[1], "MaxPool2D.strides[1]", 1);

  VerifyAttrRange(param.pool_size[0], "MaxPool2D.pool_size[0]", 
      0, dshape[hidx] + pad_h);
  VerifyAttrRange(param.pool_size[1], "MaxPool2D.pool_size[1]", 
      0, dshape[widx] + pad_w);

  dim_t tpad[2] = { param.padding[0], param.padding[0]};
  if(param.padding.ndim() == 2U){
    tpad[1] = param.padding[1];
  }

  VERIFY(tpad[0] < param.pool_size[0]);
  VERIFY(tpad[1] < param.pool_size[1]);

  //VERIFY(param.ceil_mode == false);
  if (!param.ceil_mode) {
    oshape[hidx] = ((dshape[hidx] + pad_h - param.pool_size[0]) /
                    param.strides[0]) + 1;
    oshape[widx] = ((dshape[widx] + pad_w - param.pool_size[1]) /
                    param.strides[1]) + 1;
  } else {
    oshape[hidx] = ((dshape[hidx] + pad_h - param.pool_size[0] +
                    param.strides[0] - 1) / param.strides[0]) + 1;
    int32_t min_o_h = (oshape[hidx]-1) * param.strides[0] - param.padding[0];
    VERIFY(min_o_h < dshape[hidx]);
    
    oshape[widx] = ((dshape[widx] + pad_w - param.pool_size[1] +
                    param.strides[1] - 1) / param.strides[1]) + 1;
    int32_t min_o_w = (oshape[widx]-1) * param.strides[1] - (param.padding.ndim() == 1 ? param.padding[0] : param.padding[1]);
    VERIFY(min_o_w < dshape[widx]);
  }
	CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

template <typename T>
inline bool Pool2DCorrectLayout(const NodeAttrs& attrs,
                                std::vector<Layout> *ilayouts,
                                const std::vector<Layout> *last_ilayouts,
                                std::vector<Layout> *olayouts) {
  const T &param = cvm::get<T>(attrs.parsed);
  VERIFY_EQ(ilayouts->size(), 1);
  VERIFY_EQ(last_ilayouts->size(), 1);
  VERIFY_EQ(olayouts->size(), 1);

  Layout input = (*ilayouts)[0];
  const Layout layout(param.layout);

  if (input.defined()) {
    VERIFY(input.convertible(layout)) << "Invalid input layout " << input;
    if (input.indexof('W') != layout.indexof('W') ||
        input.indexof('H') != layout.indexof('H') ||
        input.contains('w') || input.contains('h')) {
      // as long as the index doesn't change for width and height
      // pool2d can keep the input layout.
      input = layout;
    }
  } else {
    input = layout;
  }

  CVM_ASSIGN_LAYOUT(*ilayouts, 0, input);
  CVM_ASSIGN_LAYOUT(*olayouts, 0, input);

  return true;
}

CVM_REGISTER_OP(max_pool2d)
.describe(R"code(Max pooling operation for one dimensional data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, out_height, out_width)  if `layout` is `NCHW`.
           out_height and out_width are calculated as::

               out_height = floor((height+padding[0]+padding[2]-pool_size[0])/strides[0])+1
               out_width = floor((width+padding[1]+padding[3]-pool_size[1])/strides[1])+1

           where padding will be an expanded array based on number of values passed as::
               one int : all sides same padding used.
               two int : bottom, right use same as top and left.
               four int: padding width in the order of (top, left, bottom, right).

           When `ceil_mode` is `True`, ceil will be used instead of floor in this
           equation.

)code" CVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(MaxPool2DParam::__FIELDS__())
.set_attr_parser(ParamParser<MaxPool2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<MaxPool2DParam>)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", Pool2DInferShape<MaxPool2DParam>)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_attr<FCorrectLayout>("FCorrectLayout", Pool2DCorrectLayout<MaxPool2DParam>)
.set_support_level(2);

}  // namespace top
}  // namespace cvm
