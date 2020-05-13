/*!
 *  Copyright (c) 2017 by Contributors
 * \file broadcast.cc
 * \brief broadcast operator.
 */
#include <cvm/op.h>
#include <cvm/op_attr_types.h>
#include <cvm/top/tensor.h>
#include <cvm/top/nn.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {


CVMUTIL_REGISTER_PARAMETER(NonMaximumSuppressionParam);

inline bool NMSInferPrecision(
              const NodeAttrs& attrs,
              std::vector<TShape> *shapes,
              std::vector<int> *iattr,
              std::vector<int> *oattr) {
  IN_PREC_CHECK(iattr, attrs.name);

  VERIFY(iattr->at(0) <= 30)
    << "nms only supported input data precision less than 30 vs. "
    << iattr->at(0);
  (*oattr)[0] = iattr->at(0);
  return true;
}
bool NMSShape(const NodeAttrs& attrs,
              std::vector<TShape> *in_attrs,
              std::vector<TShape> *out_attrs) {
  const NonMaximumSuppressionParam& param =
    cvm::get<NonMaximumSuppressionParam>(attrs.parsed);
  VERIFY_EQ(in_attrs->size(), 2U) << "Inputs: [data, valid_count]";
  TShape dshape = in_attrs->at(0);
  TShape vshape = in_attrs->at(1);
  VERIFY_EQ(dshape.ndim(), 3U) << "Input data should be 3-D.";
  VERIFY_EQ(vshape.ndim(), 1U) << "Input valid count should be 1-D.";
  VERIFY_EQ(dshape[2], 6U) << "Data input should have shape "
    "(batch_size, num_anchors, 6).";
  VERIFY_EQ(dshape[0], vshape[0]) << "batch_size mismatch.";
  out_attrs->clear();

  VERIFY(param.coord_start == 2);
  VERIFY(param.score_index == 1);
  VERIFY(param.id_index == 0);
  VERIFY(param.iou_threshold > 0);

  VERIFY_EQ(param.return_indices, false)
    << "NonMaximumSuppressionParam only supported return_indices false vs. "
    << param.return_indices;
  VERIFY_EQ(param.invalid_to_bottom, true)
    << "NonMaximumSuppressionParam only supported invalid_to_bottom false vs. "
    << param.invalid_to_bottom;
  if (param.return_indices) {
    TShape oshape = TShape(2);
    oshape[0] = dshape[0];
    oshape[1] = dshape[1];
    CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  } else {
    CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, dshape);
  }
  return true;
}

inline bool NMSInferType(const NodeAttrs &attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  DTYPE_ASSIGN(out_attrs->at(0), in_attrs->at(0));
  return true;
}

inline bool NMSInferLayout(const NodeAttrs& attrs,
                           std::vector<Layout> *ilayouts,
                           const std::vector<Layout> *last_ilayouts,
                           std::vector<Layout> *olayouts) {
  static const Layout kNCHW("NCHW");
  VERIFY_EQ(ilayouts->size(), 2U);
  VERIFY_EQ(olayouts->size(), 1U);
  CVM_ASSIGN_LAYOUT(*ilayouts, 0, kNCHW);
  CVM_ASSIGN_LAYOUT(*ilayouts, 1, kNCHW);
  return true;
}

CVM_REGISTER_OP(non_max_suppression)
  .describe(R"doc("Non-maximum suppression."
)doc" CVM_ADD_FILELINE)
.add_alias("vision.non_max_suppression")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NonMaximumSuppressionParam>)
.set_attr<FGetAttrDict>("FGetAttrDict",
                        ParamGetAttrDict<NonMaximumSuppressionParam>)
.add_arguments(NonMaximumSuppressionParam::__FIELDS__())
.add_argument("data", "Tensor", "Input data.")
.add_argument("valid_count", "Tensor", "Number of valid anchor boxes.")
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "valid_count"};
})
.set_attr<FInferShape>("FInferShape", NMSShape)
.set_attr<FInferType>("FInferType", NMSInferType)
.set_attr<FCorrectLayout>("FCorrectLayout", NMSInferLayout)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_attr<FOpExtraSpace>("FOpExtraSpace",
    [](const NodeAttrs& attrs, 
      std::vector<TShape>* shapes,
      std::vector<int>* iprecs,
      const DLContext& ctx) -> int64_t {
    if(ctx.device_type == kDLGPU){
      TShape xshape = shapes->at(0); 
      int32_t size_offset = sizeof(int64_t) / sizeof(int32_t);
      CHECK_EQ(xshape.ndim(), 3);
      int32_t xn = (xshape[1] + size_offset - 1) / size_offset * size_offset;
      int32_t yn = size_offset;
      return (xn + yn) * size_offset;
    }
    return 0;
    })
.set_support_level(4);


CVMUTIL_REGISTER_PARAMETER(GetValidCountsParam);

bool GetValidShape(const NodeAttrs& attrs,
              std::vector<TShape> *in_attrs,
              std::vector<TShape> *out_attrs) {
  TShape shp = in_attrs->at(0);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  VERIFY(shp.ndim() == 3);
  VERIFY(shp[2] < 32 && shp[2] >= 2);
  TShape count_shape{shp[0]};
  TShape oshape(shp);
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, count_shape);
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 1, oshape);
  return true;
}

inline bool GetValidType(const NodeAttrs &attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  DTYPE_ASSIGN(out_attrs->at(0), kInt32);
  DTYPE_ASSIGN(out_attrs->at(1), in_attrs->at(0));
  return true;
}

inline bool GetValidLayout(const NodeAttrs& attrs,
                           std::vector<Layout> *ilayouts,
                           const std::vector<Layout> *last_ilayouts,
                           std::vector<Layout> *olayouts) {
  static const Layout kNCHW("NCHW");
  CHECK_EQ(ilayouts->size(), 1U);
  CHECK_EQ(olayouts->size(), 2U);
  CVM_ASSIGN_LAYOUT(*ilayouts, 0, kNCHW);
  return true;
}

inline bool GetValidInferPrecision(
              const NodeAttrs& attrs,
              std::vector<TShape> *shapes,
              std::vector<int> *iattr,
              std::vector<int> *oattr) {
  IN_PREC_CHECK(iattr, attrs.name);
  const auto& shp = shapes->at(0);
  int64_t inl = shp.Size() / shp[0];
  auto oprec1 = GetNumberPrecision(inl);
  (*oattr)[0] = oprec1;
  (*oattr)[1] = iattr->at(0);
  return true;
}

CVM_REGISTER_OP(get_valid_counts)
.describe(R"doc(Get valid count of bounding boxes given
a score threshold. Also moves valid boxes to the top of
input data.
)doc" CVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.add_argument("data", "Tensor", "Input data.")
.set_attr_parser(ParamParser<GetValidCountsParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<GetValidCountsParam>)
.add_arguments(GetValidCountsParam::__FIELDS__())
.set_attr<FInferShape>("FInferShape", GetValidShape)
.set_attr<FInferType>("FInferType", GetValidType)
.set_attr<FCorrectLayout>("FCorrectLayout", GetValidLayout)
.set_attr<FInferPrecision>("FInferPrecision", GetValidInferPrecision)
.set_attr<FOpExtraSpace>("FOpExtraSpace",
    [](const NodeAttrs& attrs, 
      std::vector<TShape>* shapes,
      std::vector<int>* iprecs,
      const DLContext& ctx) -> int64_t {
    if(ctx.device_type == kDLGPU){
      TShape shape = shapes->at(0);
      CHECK_EQ(shape.ndim(), 3U);
      return shape[0] * shape[1];
    }
    return 0;
    })
.set_support_level(4);


}
}
