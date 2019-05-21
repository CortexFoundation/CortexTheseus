/*!
 *  Copyright (c) 2017 by Contributors
 * \file matrix_op.cc
 * \brief Matrix operators
 */
#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/top/tensor.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {

CVMUTIL_REGISTER_PARAMETER(MatMulParam);

inline bool DotShape(const cvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_attrs,
                     std::vector<TShape> *out_attrs) {
  const MatMulParam& param = cvm::get<MatMulParam>(attrs.parsed);
  VERIFY_EQ(in_attrs->size(), 2U);
  VERIFY_EQ(out_attrs->size(), 1U);
  TShape lshape = (*in_attrs)[0];
  TShape rshape = (*in_attrs)[1];

  if (lshape.ndim() == 1)  lshape = TShape{1, lshape[0]};
  if (rshape.ndim() == 1) rshape = TShape{1, rshape[0]};

  if (param.transpose_a) std::reverse(lshape.begin(), lshape.end());
  if (param.transpose_b) std::reverse(rshape.begin(), rshape.end());

  VERIFY_EQ(lshape[lshape.ndim() - 1], rshape[0])
    << "dot shape inconsistent: " << lshape << " X " << rshape;

  TShape oshape(lshape.ndim() + rshape.ndim() - 2);
  for (uint32_t i = 0; i < lshape.ndim() - 1; i++) oshape[i] = lshape[i];
  for (uint32_t i = 1; i < rshape.ndim(); i++) oshape[i + lshape.ndim() - 2] = rshape[i];

  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  return true;
}

inline bool DotCorrectLayout(const NodeAttrs& attrs,
                             std::vector<Layout> *ilayouts,
                             const std::vector<Layout> *last_ilayouts,
                             std::vector<Layout> *olayouts) {
  const MatMulParam& param = cvm::get<MatMulParam>(attrs.parsed);
  VERIFY_EQ(ilayouts->size(), 2U);
  VERIFY_EQ(olayouts->size(), 1U);
  const Layout& lhs = last_ilayouts->at(0).defined() ? last_ilayouts->at(0)
                                                     : ilayouts->at(0);
  const Layout& rhs = last_ilayouts->at(1).defined() ? last_ilayouts->at(1)
                                                     : ilayouts->at(1);
  CVM_ASSIGN_LAYOUT(*ilayouts, 0, lhs);
  CVM_ASSIGN_LAYOUT(*ilayouts, 1, rhs);

  if (lhs.ndim() > 1 && rhs.ndim() > 1) {
    // concat lhs and rhs layout
    const Layout& lhs_out = param.transpose_a ? lhs.reverse() : lhs;
    const Layout& rhs_out = param.transpose_b ? rhs.reverse() : rhs;
    Layout out = lhs_out.sublayout(0, lhs_out.ndim()-1) +
        rhs_out.sublayout(1, rhs_out.ndim()-1);
    CVM_ASSIGN_LAYOUT(*olayouts, 0, out);
  }
  return true;
}

CVM_REGISTER_OP(matmul)
.describe(R"doc(Matrix multiplication of two arrays.

``dot``'s behavior depends on the input array dimensions:

- 1-D arrays: inner product of vectors
- 2-D arrays: matrix multiplication
- N-D arrays: a sum product over the last axis of the first input and the first
  axis of the second input

  For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the
  result array will have shape `(n,m,r,s)`. It is computed by::

    dot(x,y) = sum(x[i,j,:]*y[:,a,b])

)doc" CVM_ADD_FILELINE)
.set_support_level(1)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<MatMulParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<MatMulParam>)
.add_arguments(MatMulParam::__FIELDS__())
.add_argument("lhs", "NDArray-or-Symbol", "The first input")
.add_argument("rhs", "NDArray-or-Symbol", "The second input")
.set_attr<FInferShape>("FInferShape", DotShape)
.set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferPrecision>("FInferPrecision",
  [](const NodeAttrs& attrs,
   std::vector<TShape>* shapes,
   std::vector<int>* iattr,
   std::vector<int>* oattr) -> bool {
  auto& param = cvm::get<MatMulParam>(attrs.parsed);
  int prec = iattr->at(0) + iattr->at(1);
  if (param.transpose_a) {
    prec += CORTEX_LOG2(shapes->at(0)[0]) + 1;
  } else {
    prec += CORTEX_LOG2(shapes->at(0)[1]) + 1;
  }
  (*oattr)[0] = prec;
  return true;
})
.set_attr<FCorrectLayout>("FCorrectLayout", DotCorrectLayout);

}  // namespace top
}  // namespace cvm
