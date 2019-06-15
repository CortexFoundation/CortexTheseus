/*!
 *  Copyright (c) 2017 by Contributors
 * \file convolution.cc
 * \brief Convolution operators
 */
#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/layout.h>
#include <cvm/op_attr_types.h>
#include <cvm/top/nn.h>
#include "nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {

// conv2d
CVMUTIL_REGISTER_PARAMETER(Conv2DParam);

inline bool Conv2DInferShape(const cvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_shape,
                             std::vector<TShape>* out_shape) {
  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const Conv2DParam& param = cvm::get<Conv2DParam>(attrs.parsed);

  VERIFY_EQ(param.layout, "NCHW")
    << "Conv2D only supported layout: NCHW vs. " << param.layout;
  VERIFY_EQ(param.kernel_layout, "OIHW")
    << "Conv2D only supported kernel layout: OIHW vs. " << param.kernel_layout;
  const Layout in_layout(param.layout);
  const Layout kernel_layout(param.kernel_layout);
  // VERIFY(in_layout.convertible(kNCHW))
  //   << "Conv only support input layouts that are convertible from NCHW."
  //   << " But got " << in_layout;
  // VERIFY(kernel_layout.convertible(kOIHW))
  //   << "Conv only support kernel layouts that are convertible from OIHW."
  //   << " But got "<< kernel_layout;

  Layout out_layout(param.out_layout);
  if (!out_layout.defined()) out_layout = in_layout;
  VERIFY_EQ(out_layout.name(), "NCHW");
  // VERIFY(out_layout.convertible(kNCHW))
  //   << "Conv only support output layouts that are convertible from NCHW."
  //   << " But got " << out_layout;

  if (param.use_bias) {
    VERIFY_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    VERIFY_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  VERIFY_EQ(out_shape->size(), 1U);

  TShape dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  // dshape = ConvertLayout(dshape, in_layout, kNCHW);

  VERIFY_EQ(dshape.ndim(), 4U) << "Input data should be 4D";
  VERIFY_EQ(param.kernel_size.ndim(), 2U);
  VERIFY_EQ(param.strides.ndim(), 2U)
      << "incorrect stride size: " << param.strides;
  VERIFY_EQ(param.dilation.ndim(), 2U)
      << "incorrect dilate size: " << param.dilation;
  VERIFY_EQ(dshape[1] % param.groups, 0U)
      << "input channels must divide group size";
  VERIFY_EQ(param.channels % param.groups, 0U)
      << "output channels must divide group size";
  TShape outshape = out_shape->at(0);
  bool check_groups = ((dshape[1] == param.groups && outshape[1] == param.groups) || (param.groups == 1));
  if (not check_groups) {
    VERIFY(false)
      << "Conv2D only supported groups (1 or in_channels " << param.channels
      << ") vs. " << param.groups;
  }

  TShape wshape({param.channels,
                 dshape[1] / param.groups,
                 param.kernel_size[0],
                 param.kernel_size[1]});

  // wshape = ConvertLayout(wshape, kOIHW, kernel_layout);

  if (in_shape->at(Conv2DParam::kWeight).ndim() == 0) {
    CVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DParam::kWeight, wshape);
  }
  if (param.use_bias) {
    static const Layout default_bias_layout("C");
    TShape bias_shape({param.channels});
    auto oc_block = out_layout.subsizeof('C');
    if (oc_block > 0) {
      size_t split_axis = (out_layout.indexof('C') < out_layout.indexof('c')) ? 1 : 0;
      bias_shape = ConvertLayout(bias_shape, default_bias_layout,
                                 default_bias_layout.split('C', split_axis, oc_block));
    }
    CVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DParam::kBias, bias_shape);
  }
  // dilation
  dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
  dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
  TShape oshape({dshape[0], param.channels, 0, 0});
  if (dshape[2] != 0) {
    oshape[2] = (dshape[2] + param.padding[0] * 2 - dilated_ksize_y) / param.strides[0] + 1;
  }
  if (dshape[3] != 0) {
    oshape[3] = (dshape[3] + param.padding[1] * 2 - dilated_ksize_x) / param.strides[1] + 1;
  }
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, ConvertLayout(oshape, kNCHW, out_layout));
  // Perform incomplete shape inference. Fill in the missing values in data shape.
  // 1) We can always fill in the batch_size.
  // 2) We can back-calculate the input height/width if the corresponding stride is 1.
  // oshape = ConvertLayout((*out_shape)[0], out_layout, kNCHW);
  dshape[0] = oshape[0];
  if (oshape[2] && param.strides[0] == 1) {
    dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param.padding[0];
  }
  if (oshape[3] && param.strides[1] == 1) {
    dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param.padding[1];
  }
  CVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, Conv2DParam::kData,
                          ConvertLayout(dshape, kNCHW, in_layout));
  // Check whether the kernel sizes are valid
  if (dshape[2] != 0) {
    VERIFY_LE(dilated_ksize_y, dshape[2] + 2 * param.padding[0])
      << "kernel size exceed input";
  }
  if (dshape[3] != 0) {
    VERIFY_LE(dilated_ksize_x, dshape[3] + 2 * param.padding[1])
        << "kernel size exceed input";
  }
  return true;
}


template <typename PARAM>
inline bool Conv2DInferType(const cvm::NodeAttrs& attrs,
                            std::vector<int>* in_type,
                            std::vector<int>* out_type) {
  const PARAM& param = cvm::get<PARAM>(attrs.parsed);
  if (param.use_bias) {
    VERIFY_EQ(in_type->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    VERIFY_EQ(in_type->size(), 2U) << "Input:[data, weight]";
  }
  VERIFY_EQ(out_type->size(), 1U);
  if (param.out_dtype != -1) {
    VERIFY(!type_is_none((*in_type)[0]));
    for (size_t i = 1; i < in_type->size(); ++i) {
      CVM_ASSIGN_INPUT_TYPE(attrs, *in_type, i, (*in_type)[0]);
    }
    CVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, param.out_dtype);
  } else {
    ElemwiseType<-1, 1>(attrs, in_type, out_type);
  }
  return true;
}
template <typename PARAM>
inline bool Conv2DInferPrecision(const NodeAttrs& attrs,
		                             std::vector<TShape>* shapes,
																 std::vector<int>* iattr,
																 std::vector<int>* oattr) {
  IN_PREC_CHECK(iattr, attrs.name);
  VERIFY_LE(iattr->at(0), 8)
    << "Conv2D " << attrs.name
    << " input must be INT8 vs. INT" << iattr->at(0);
  VERIFY_LE(iattr->at(1), 8)
    << "Conv2D " << attrs.name
    << " weight must be INT8 vs. INT" << iattr->at(1);
  if (shapes->size() == 0 || shapes->at(0)[1] == 0)
      return false;
  const TShape& wshp = shapes->at(1);
  VERIFY_EQ(wshp.ndim(), 4);
  int64_t max_size = wshp.Size() / wshp[0];
  int oprec = iattr->at(0) + iattr->at(1);
  oprec += GetBit(max_size);

  const PARAM& param = cvm::get<PARAM>(attrs.parsed);
  if (param.use_bias) {
    int bias_prec = iattr->at(2);
    oprec = std::max(oprec, bias_prec) + 1;
  }
  (*oattr)[0] = oprec;
  return true;
}


template<typename PARAM>
inline bool Conv2DCorrectLayout(const NodeAttrs& attrs,
                                std::vector<Layout> *ilayouts,
                                const std::vector<Layout> *last_ilayouts,
                                std::vector<Layout> *olayouts) {
  const PARAM& param = cvm::get<PARAM>(attrs.parsed);

  const Layout in_layout(param.layout);
  Layout out_layout(param.out_layout);
  if (!out_layout.defined()) out_layout = in_layout;

  const Layout kernel_layout(param.kernel_layout);
  if (param.use_bias) {
    VERIFY_EQ(ilayouts->size(), 3U) << "Input:[data, weight, bias]";
    CVM_ASSIGN_LAYOUT(*ilayouts, 0, in_layout);
    CVM_ASSIGN_LAYOUT(*ilayouts, 1, kernel_layout);
    // automatically decide bias layout
    Layout bias_layout("C");
    auto oc_block = out_layout.subsizeof('C');
    if (oc_block > 0) {
      size_t split_axis = (out_layout.indexof('C') < out_layout.indexof('c')) ? 1 : 0;
      bias_layout = bias_layout.split('C', split_axis, oc_block);
    }
    CVM_ASSIGN_LAYOUT(*ilayouts, 2, bias_layout);
  } else {
    VERIFY_EQ(ilayouts->size(), 2U) << "Input:[data, weight]";
    CVM_ASSIGN_LAYOUT(*ilayouts, 0, in_layout);
    CVM_ASSIGN_LAYOUT(*ilayouts, 1, kernel_layout);
  }

  VERIFY_EQ(olayouts->size(), 1U);
  CVM_ASSIGN_LAYOUT(*olayouts, 0, out_layout);

  return true;
}

CVM_REGISTER_OP(conv2d)
.describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" CVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(Conv2DParam::__FIELDS__())
.set_attr_parser(ParamParser<Conv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<Conv2DParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<Conv2DParam>)
.set_attr<FInferShape>("FInferShape", Conv2DInferShape)
.set_attr<FInferType>("FInferType", Conv2DInferType<Conv2DParam>)
.set_attr<FInferPrecision>("FInferPrecision", Conv2DInferPrecision<Conv2DParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", Conv2DCorrectLayout<Conv2DParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<Conv2DParam>)
.set_support_level(2);

}  // namespace top
}  // namespace cvm
