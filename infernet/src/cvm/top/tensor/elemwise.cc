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

// undefined op
CVM_REGISTER_ELEMWISE_UNARY_OP(__undef__)
.describe(R"code(undefined op.

Used to produce invalide node during optimization.

)code" CVM_ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(0);

// abs
CVM_REGISTER_ELEMWISE_UNARY_OP(abs)
.describe(R"code(Take absolute value of elements of the input.
)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", NonScalePrecision)
.set_support_level(3);

// sigmoid
// CVM_REGISTER_ELEMWISE_UNARY_OP(sigmoid)
// .describe(R"code(Computes sigmoid.
// 
// .. math::
//   Y = 1 / (1 + exp(-X))
// 
// )code" CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwiseSamePrecision)
// .set_support_level(1);

// tanh
// CVM_REGISTER_ELEMWISE_UNARY_OP(tanh)
// .describe(R"code(Computes hyperbolic tangent.
// 
// .. math::
//    Y = sinh(X) / cosh(X)
// 
// )code" CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwiseSamePrecision)
// .set_support_level(1);

// log2
CVM_REGISTER_ELEMWISE_UNARY_OP(log2)
.describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log2(x)

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<5>)
.set_support_level(1);

// log
// CVM_REGISTER_ELEMWISE_UNARY_OP(log)
// .describe(R"code(Returns the log input array, computed element-wise.
// 
// .. math::
//    log(x)
// 
// )code" CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<5>)
// .set_support_level(1);

// sqrt
// CVM_REGISTER_ELEMWISE_UNARY_OP(sqrt)
// .describe(R"code(Returns the sqrt input array, computed element-wise.
// 
// .. math::
//    \sqrt(x)
// 
// )code" CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", 
//     [](const NodeAttrs& attrs,
//       std::vector<TShape>* shapes,
//       std::vector<int>* iattr,
//       std::vector<int>* oattr) -> bool {
//       if (iattr->size() != oattr->size()) {
//         return false;
//       }
//       for (int i = 0; i < oattr->size(); ++i) {
//         (*oattr)[i] = (iattr->at(i) + 1) >> 1;
//       }
//       return true;
//     })
// .set_support_level(1);

// binary ops

CVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_add)
.describe(R"code(Element-wise add

)code")
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePlusonePrecision)
.set_support_level(1);

CVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_sub)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePlusonePrecision)
.describe(R"code(Element-wise substraction

)code"  CVM_ADD_FILELINE)
.set_support_level(1);

CVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_mul)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseSumPrecision)
.describe(R"code(Element-wise multiplication

)code"  CVM_ADD_FILELINE)
.set_support_level(1);

CVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_div)
.describe(R"code(Element-wise division

)code"  CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseFirstPrecision)
.set_support_level(1);

CVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_mod)
  .describe(R"code(Element-wise modulo

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseFirstPrecision)
.set_support_level(1);

// logical
// CVM_REGISTER_ELEMWISE_BINARY_OP(logical_and)
// .describe(R"code(Elementwise compute the logical AND
// 
// )code")
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>)
// .set_support_level(4);
// 
// CVM_REGISTER_ELEMWISE_BINARY_OP(logical_or)
// .describe(R"code(Elementwise compute the logical OR
// 
// )code")
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>)
// .set_support_level(4);

// negative
CVM_REGISTER_ELEMWISE_UNARY_OP(negative)
.describe(R"code(Elemenwise numeric negative

)code"  CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseSamePrecision)
.set_support_level(3);

// logical NOT
// CVM_REGISTER_ELEMWISE_UNARY_OP(logical_not)
// .describe(R"code(Elementwise compute the logical NOT
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>)
// .set_support_level(4);

// copy
// CVM_REGISTER_ELEMWISE_UNARY_OP(copy)
// .describe(R"code(Copy tensor to another one.
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwiseSamePrecision)
// .set_support_level(3);
// 
// CVMUTIL_REGISTER_PARAMETER(InitOpParam);
// CVMUTIL_REGISTER_PARAMETER(InitOpWithScalarParam);
// CVMUTIL_REGISTER_PARAMETER(FillValueParam);

// full
// CVM_REGISTER_INIT_OP(full)
// .describe(R"code(Fill array with scalar value
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr_parser(ParamParser<InitOpWithScalarParam>)
// .set_attr<FGetAttrDict>(
//   "FGetAttrDict", ParamGetAttrDict<InitOpWithScalarParam>)
// .add_arguments(InitOpWithScalarParam::__FIELDS__())
// .set_attr<FInferShape>("FInferShape", ZeroShape<InitOpWithScalarParam>)
// .set_attr<FInferType>("FInferType", ZeroType<InitOpWithScalarParam>)
// .set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
// .set_attr<FInferPrecision>("FInferPrecision",
//     [](const NodeAttrs& attrs,
//       std::vector<TShape>* shapes,
//       std::vector<int>* iattr,
//       std::vector<int>* oattr) -> bool {
//     auto& param = cvm::get<InitOpWithScalarParam>(attrs.parsed);
//     auto fill_value = param.fill_value;
//     int prec = CORTEX_LOG2(fill_value);
//     if (oattr->size() == 0) return false;
//     (*oattr)[0] = prec;
//      return true;
//   })
// .set_support_level(4);
// 
// CVM_REGISTER_INIT_OP(zeros)
// .describe(R"code(Fill target with zeros
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr_parser(ParamParser<InitOpParam>)
// .set_attr<FGetAttrDict>(
//   "FGetAttrDict", ParamGetAttrDict<InitOpParam>)
// .add_arguments(InitOpParam::__FIELDS__())
// .set_attr<FInferShape>("FInferShape", ZeroShape<InitOpParam>)
// .set_attr<FInferType>("FInferType", ZeroType<InitOpParam>)
// .set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>)
// .set_support_level(4);
// 
// CVM_REGISTER_INIT_OP(ones)
// .describe(R"code(Fill target with ones
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr_parser(ParamParser<InitOpParam>)
// .set_attr<FGetAttrDict>(
//   "FGetAttrDict", ParamGetAttrDict<InitOpParam>)
// .add_arguments(InitOpParam::__FIELDS__())
// .set_attr<FInferShape>("FInferShape", ZeroShape<InitOpParam>)
// .set_attr<FInferType>("FInferType", ZeroType<InitOpParam>)
// .set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>)
// .set_support_level(4);

// full_like
// CVM_REGISTER_INIT_LIKE_OP(full_like)
// .describe(R"code(Return an scalar value array with the same shape and type
// as the input array
// 
// )code"  CVM_ADD_FILELINE)
// .add_arguments(FillValueParam::__FIELDS__())
// .set_attr_parser(ParamParser<FillValueParam>)
// .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<FillValueParam>)
// .set_attr<FInferPrecision>("FInferPrecision",
//     [](const NodeAttrs& attrs,
//       std::vector<TShape>* shapes,
//       std::vector<int>* iattr,
//       std::vector<int>* oattr) -> bool {
//     auto& param = cvm::get<FillValueParam>(attrs.parsed);
//     auto fill_value = param.fill_value;
//     int prec = CORTEX_LOG2(fill_value);
//     if (oattr->size() == 0) return false;
//     (*oattr)[0] = prec;
//      return true;
//   })
// .set_support_level(4);

// CVM_REGISTER_INIT_LIKE_OP(zeros_like)
// .describe(R"code(Return an array of zeros with the same shape and type
// as the input array.
// 
// )code")
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>)
// .set_support_level(4);
// 
// CVM_REGISTER_INIT_LIKE_OP(ones_like)
// .describe(R"code(Return an array of ones with the same shape and type
// as the input array.
// 
// )code")
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>)
// .set_support_level(4);

// unary scalar op
// CVMUTIL_REGISTER_PARAMETER(ScalarParam);
// 
// #define CVM_REGISTER_ELEMWISE_BINARY_SCALAR(op)                        \
//   CVM_REGISTER_ELEMWISE_UNARY_OP(op)                                   \
//   .add_arguments(ScalarParam::__FIELDS__())                             \
//   .set_attr_parser(ParamParser<ScalarParam>)                            \
//   .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ScalarParam>)
// 
// inline bool AddScalarInferPrecision(const NodeAttrs& attrs,
//       std::vector<TShape>* shapes,
//       std::vector<int>* iattr,
//       std::vector<int>* oattr) {
//   auto& param = cvm::get<ScalarParam>(attrs.parsed);
//   int prec = CORTEX_LOG2(param.scalar);
//   (*oattr)[0] = std::max(prec, iattr->at(0)) + 1;
//   return true;
// }
// 
// CVM_REGISTER_ELEMWISE_BINARY_SCALAR(__add_scalar__)
// .describe(R"code(Tensor add scalar
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", AddScalarInferPrecision)
// .set_support_level(3);
// 
// CVM_REGISTER_ELEMWISE_BINARY_SCALAR(__sub_scalar__)
// .describe(R"code(Tensor substract scalar
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", AddScalarInferPrecision)
// .set_support_level(3);
// 
// CVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rsub_scalar__)
// .describe(R"code(scalar substract Tensor
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", AddScalarInferPrecision)
// .set_support_level(3);
// 
// CVM_REGISTER_ELEMWISE_BINARY_SCALAR(__lshift_scalar__)
// .describe(R"code(Tensor left shift by scalar
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision",
//   [](const NodeAttrs& attrs,
//    std::vector<TShape>* shapes,
//    std::vector<int>* iattr,
//    std::vector<int>* oattr) -> bool {
//   auto& param = cvm::get<ScalarParam>(attrs.parsed);
//   (*oattr)[0] = param.scalar + iattr->at(0);
//   return true;
// })
// .set_support_level(3);
// 
// CVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rshift_scalar__)
// .describe(R"code(Tensor right shift by scalar
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision",
//   [](const NodeAttrs& attrs,
//    std::vector<TShape>* shapes,
//    std::vector<int>* iattr,
//    std::vector<int>* oattr) -> bool {
//   auto& param = cvm::get<ScalarParam>(attrs.parsed);
//   (*oattr)[0] = iattr->at(0) - param.scalar;
//   return true;
// })
// .set_support_level(3);
// 
// CVM_REGISTER_ELEMWISE_BINARY_SCALAR(__mul_scalar__)
// .describe(R"code(Tensor multiplies scalar
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision",
//   [](const NodeAttrs& attrs,
//    std::vector<TShape>* shapes,
//    std::vector<int>* iattr,
//    std::vector<int>* oattr) -> bool {
//   auto& param = cvm::get<ScalarParam>(attrs.parsed);
//   (*oattr)[0] = CORTEX_LOG2(param.scalar) + iattr->at(0);
//   return true;
// })
// .set_support_level(3);
// 
// CVM_REGISTER_ELEMWISE_BINARY_SCALAR(__div_scalar__)
// .describe(R"code(Tensor divides scalar
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision",
//   [](const NodeAttrs& attrs,
//    std::vector<TShape>* shapes,
//    std::vector<int>* iattr,
//    std::vector<int>* oattr) -> bool {
//   auto& param = cvm::get<ScalarParam>(attrs.parsed);
//   (*oattr)[0] = iattr->at(0) - CORTEX_LOG2(param.scalar);
//   return true;
// })
// .set_support_level(3);
// 
// CVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rdiv_scalar__)
// .describe(R"code(scalar divides Tensor
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision",
//   [](const NodeAttrs& attrs,
//    std::vector<TShape>* shapes,
//    std::vector<int>* iattr,
//    std::vector<int>* oattr) -> bool {
//   auto& param = cvm::get<ScalarParam>(attrs.parsed);
//   (*oattr)[0] = CORTEX_LOG2(param.scalar);
//   return true;
// })
// .set_support_level(3);
// 
// CVM_REGISTER_ELEMWISE_BINARY_SCALAR(__pow_scalar__)
// .describe(R"code(Tensor power scalar
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision",
//   [](const NodeAttrs& attrs,
//    std::vector<TShape>* shapes,
//    std::vector<int>* iattr,
//    std::vector<int>* oattr) -> bool {
//   auto& param = cvm::get<ScalarParam>(attrs.parsed);
//   (*oattr)[0] = param.scalar * iattr->at(0);
//   return true;
// })
// .set_support_level(3);

// CVMUTIL_REGISTER_PARAMETER(ElementWiseReduceParam);
// 
// inline bool ElemwiseSumInferPrecision(const NodeAttrs& attrs,
//    std::vector<TShape>* shapes,
//    std::vector<int>* iattr,
//    std::vector<int>* oattr) {
//   uint64_t sum = 0;
//   for (auto x : *iattr) {
//     if (x > 32) return false;
//     sum += (1ull << x);
//   }
//   if (sum >= (1ull << 32)) return false;
//   (*oattr)[0] = CORTEX_LOG2(sum);
//   return true;
// }
// 
// CVM_REGISTER_ELEMWISE_REDUCE_OP(elemwise_sum)
// .describe(R"code(Adds all input arguments element-wise.
// 
// )code"  CVM_ADD_FILELINE)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwiseSumInferPrecision)
// .set_support_level(4);

// CVMUTIL_REGISTER_PARAMETER(IndicatorParam);

// indicator function
// CVM_REGISTER_INDICATOR_OP(greater)
// .describe(R"code(Greater function that returns a mask tensor
// with 1.0 if (left > right), otherwise 0.0 element-wise.
// 
// )code" CVM_ADD_FILELINE)
// .add_argument("lhs", "Tensor", "First input")
// .add_argument("rhs", "Tensor", "Second input")
// .set_num_inputs(2)
// .set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>)
// .set_support_level(4);


// CVM_REGISTER_INDICATOR_OP(less)
//   .describe(R"code(Less function that returns a mask tensor
// with 1.0 if (left < right), otherwise 0.0 element-wise.
// 
// )code" CVM_ADD_FILELINE)
// .add_argument("lhs", "Tensor", "First input")
// .add_argument("rhs", "Tensor", "Second input")
// .set_num_inputs(2)
// .set_attr<cvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
// .set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>)
// .set_support_level(4);

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
  auto& param = cvm::get<ClipParam>(attrs.parsed);
  (*oattr)[0] = CORTEX_LOG2(std::max(param.a_max, -param.a_min + 1));
  return true;
})
.set_attr<cvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)

.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(ClipParam::__FIELDS__())
.set_support_level(4);

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
  oattr->assign(0, param.precision);
  return true;
})
.add_argument("data", "Tensor", "input")
.add_arguments(CVMClipParam::__FIELDS__())
.set_support_level(4);

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
  if (iattr->at(0) + param.shift_bit > 32) return false;
  (*oattr)[0] = param.precision;
  return true;
})
.add_argument("data", "Tensor", "input")
.add_arguments(CVMLeftShiftParam::__FIELDS__())
.set_support_level(4);

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
  oattr->assign(0, param.precision);
  return true;
})
.add_argument("data", "Tensor", "input")
.add_arguments(CVMRightShiftParam::__FIELDS__())
.set_support_level(4);



}  // namespace top
}  // namespace cvm
