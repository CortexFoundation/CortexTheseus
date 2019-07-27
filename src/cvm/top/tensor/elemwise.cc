/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise.cc
 * \brief Elemenwise operators
 */
#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/op_attr_types.h>
#include <cvm/top/tensor.h>
#include <cmath>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {

// abs
CVM_REGISTER_ELEMWISE_UNARY_OP(abs)
.describe(R"code(Take absolute value of elements of the input.
)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_support_level(3);

// cvm_precision
CVM_REGISTER_ELEMWISE_UNARY_OP(cvm_precision)
.describe(R"code(Returns the precision of input array, computed element-wise.

.. math::
   precision(x) = ceil(log2(x+1)) x > 0;
   precision(x) = 1 x = 1;
   precision(x) = precision(abs(x)) x < 0;

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", MaxPrecision<6>)
.set_support_level(1);

// binary ops
CVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_add)
.describe(R"code(Element-wise add

)code")
.set_attr<FInferPrecision>("FInferPrecision", BinaryPlusPrecision)
.set_support_level(1);

CVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_sub)
.set_attr<FInferPrecision>("FInferPrecision", BinaryPlusPrecision)
.describe(R"code(Element-wise substraction

)code"  CVM_ADD_FILELINE)
.set_support_level(1);

// negative
CVM_REGISTER_ELEMWISE_UNARY_OP(negative)
.describe(R"code(Elemenwise numeric negative

)code"  CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_support_level(3);

CVMUTIL_REGISTER_PARAMETER(ClipParam);

CVM_REGISTER_OP(clip)
.describe(R"doc(Clips (limits) the values in an array.
Given an interval, values outside the interval are clipped to the interval edges.
Clipping ``x`` between `a_min` and `a_x` would be::
   clip(x, a_min, a_max) = max(min(x, a_max), a_min))
Example::
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
)doc" CVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ClipParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ClipParam>)
.set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<cvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision",
  [](const NodeAttrs& attrs,
   std::vector<TShape>* shapes,
   std::vector<int>* iattr,
   std::vector<int>* oattr) -> bool {
  IN_PREC_CHECK(iattr, attrs.name);
  auto& param = cvm::get<ClipParam>(attrs.parsed);
  int64_t range = std::max(std::abs(param.a_max), std::abs(param.a_min));
  (*oattr)[0] = GetBit(range) + 1;
  return true;
})
.set_attr<cvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(ClipParam::__FIELDS__())
.set_support_level(4);

// cvm_clip
CVMUTIL_REGISTER_PARAMETER(CVMClipParam);

CVM_REGISTER_OP(cvm_clip)
.describe(R"doc(CVM clip input with precision.

.. math::
  range = 2 ** (precision - (is_sign ? 1 : 0)) - 1
  a_min = is_sign ? -range : 0
  a_max = range
  Y = clip(X, a_min=a_min, a_max=a_max)

Example::

  data = [275, 157, -23, -168, -275]

  cvm_clip(data, precision=8, is_sign=True)
  [127, 127, -23, -127, -127]

  cvm_clip(data, precision=8, is_sign=False)
  [255, 157, 0, 0, 0]
)doc" CVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CVMClipParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<CVMClipParam>)
.set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<cvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<cvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision",
  [](const NodeAttrs& attrs,
     std::vector<TShape>* shapes,
     std::vector<int>* iattr,
     std::vector<int>* oattr) -> bool {
  IN_PREC_CHECK(iattr, attrs.name);
  auto& param = cvm::get<CVMClipParam>(attrs.parsed);
  VerifyAttrRange(param.precision, "cvm_clip.precision", 1, 33);
  (*oattr)[0] = param.precision;
  return true;
})
.add_argument("data", "Tensor", "input")
.add_arguments(CVMClipParam::__FIELDS__())
.set_support_level(4);

// cvm_left_shift
CVMUTIL_REGISTER_PARAMETER(CVMLeftShiftParam);

CVM_REGISTER_OP(cvm_left_shift)
.describe(R"code(CVM left shift with precision-aware clip.

.. math::
  assert shift_bit > 0
  tmp = X << shift_bit
  Y = cvm_clip(tmp, precision)
)code" CVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CVMLeftShiftParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<CVMLeftShiftParam>)
.set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<cvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<cvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision",
  [](const NodeAttrs& attrs,
   std::vector<TShape>* shapes,
   std::vector<int>* iattr,
   std::vector<int>* oattr) -> bool {
  IN_PREC_CHECK(iattr, attrs.name);
  auto& param = cvm::get<CVMLeftShiftParam>(attrs.parsed);
  VerifyAttrRange(param.precision, "cvm_left_shift.precision", 1, 33);
  VerifyAttrRange(param.shift_bit, "cvm_left_shift.shift_bit", 1, 33);
  if (iattr->at(0) + param.shift_bit > 32) return false;
  (*oattr)[0] = param.precision;
  return true;
})
.add_argument("data", "Tensor", "input")
.add_arguments(CVMLeftShiftParam::__FIELDS__())
.set_support_level(4);

// cvm_right_shift
CVMUTIL_REGISTER_PARAMETER(CVMRightShiftParam);

CVM_REGISTER_OP(cvm_right_shift)
.describe(R"code(CVM right shift with precision-aware clip.

The right shift is equal to float number round divide operator,
which means to implement via tricky equation.

.. math::
  assert shift_bit > 0
  tmp = X >> (shift_bit - 1)
  tmp = tmp + 1
  tmp = tmp >> 1
  Y = cvm_clip(tmp, precision)
)code" CVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CVMRightShiftParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<CVMRightShiftParam>)
.set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<cvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<cvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision",
  [](const NodeAttrs& attrs,
   std::vector<TShape>* shapes,
   std::vector<int>* iattr,
   std::vector<int>* oattr) -> bool {
  IN_PREC_CHECK(iattr, attrs.name);
  auto& param = cvm::get<CVMRightShiftParam>(attrs.parsed);
  VerifyAttrRange(param.precision, "cvm_right_shift.precision", 1, 33);
  VerifyAttrRange(param.shift_bit, "cvm_right_shift.shift_bit", 1, 33);
  (*oattr)[0] = param.precision;
  return true;
})
.add_argument("data", "Tensor", "input")
.add_arguments(CVMRightShiftParam::__FIELDS__())
.set_support_level(4);



}  // namespace top
}  // namespace cvm
