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

CVMUTIL_REGISTER_PARAMETER(UpSamplingParam);

inline bool UpSamplingInferShape(const cvm::NodeAttrs& attrs,
                                 std::vector<TShape>* in_shape,
                                 std::vector<TShape>* out_shape) {
  const UpSamplingParam& param = cvm::get<UpSamplingParam>(attrs.parsed);
  VERIFY_EQ(in_shape->size(), 1U);
  VERIFY_EQ(out_shape->size(), 1U);
  TShape dshape = (*in_shape)[0];
  if (dshape.ndim() ==  0) return false;

  VERIFY_EQ(dshape.ndim(), 4)
    << "dimension should be 4D, Got: " << dshape;
  VERIFY_EQ(param.method, "NEAREST_NEIGHBOR") 
    << "only accept method = NEAREST_NEIGHBOR ";
  VerifyAttrRange(param.scale, "UpSampling.scale", 1);
  VERIFY_EQ(param.layout, "NCHW")
    << "UpSampling only supported NCHW layout vs. " << param.layout;
  TShape oshape = dshape;
  oshape[2] = oshape[2] * param.scale;
  oshape[3] = oshape[3] * param.scale;
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);

  return true;
}

inline bool UpsamplingLayout(const NodeAttrs& attrs,
                             std::vector<Layout> *in_layouts,
                             const std::vector<Layout> *last_in_layouts,
                             std::vector<Layout> *out_layouts) {
  const UpSamplingParam& param = cvm::get<UpSamplingParam>(attrs.parsed);
  VERIFY_EQ(in_layouts->size(), 1U);
  VERIFY_EQ(out_layouts->size(), 1U);
  const Layout layout(param.layout);
  CVM_ASSIGN_LAYOUT(*in_layouts, 0, layout);
  CVM_ASSIGN_LAYOUT(*out_layouts, 0, layout);
  return true;
}

CVM_REGISTER_OP(upsampling)
.describe(R"(Perform upsampling to input array with nearest neighbour or bilinear interpolation.

- **data**: data is 4D array of shape
            (batch_size, channels, in_height, in_width) for NCHW
            (batch_size, in_height, in_width, channels) for NHWC

- **out**: Output is 4D array of shape
           for layout NCHW
           (batch_size, channels, in_height*scale, in_width*scale)

           for layout NHWC
           (batch_size, in_height*scale, in_width*scale, channels)

)" CVM_ADD_FILELINE)
.add_argument("data", "4D Tensor", "Input data.")
.add_arguments(UpSamplingParam::__FIELDS__())
.set_attr_parser(ParamParser<UpSamplingParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<UpSamplingParam>)
.set_attr<FInferShape>("FInferShape", UpSamplingInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", UpsamplingLayout)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_num_outputs(1)
.set_num_inputs(1)
.set_support_level(2);

}  // namespace top
}  // namespace cvm
