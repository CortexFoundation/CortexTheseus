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

// broadcast_to
CVMUTIL_REGISTER_PARAMETER(BroadcastToParam);

inline bool BroadcastToInferShape(const NodeAttrs& attrs,
                                  std::vector<TShape>* in_attrs,
                                  std::vector<TShape>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 1U);
  VERIFY_EQ(out_attrs->size(), 1U);
  const TShape& ishape = (*in_attrs)[0];
  if (ishape.ndim() == 0) return false;

  const BroadcastToParam& param = cvm::get<BroadcastToParam>(attrs.parsed);
  VERIFY_EQ(ishape.ndim(), param.shape.ndim())
      << "Operand of shape " << ishape
      << " cannot be broadcasted to " << param.shape;
  TShape oshape = param.shape;
  for (dim_t i = 0; i < ishape.ndim(); ++i) {
    if (oshape[i] != 0) {
      VERIFY(ishape[i] == oshape[i] || ishape[i] == 1)
        << "Array cannot be broadcasted from " <<
          ishape << " to " << param.shape;
    } else {
      oshape[i] = ishape[i];
    }
  }
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  return true;
}

CVM_REGISTER_OP(broadcast_to)
.describe(R"code(Broadcasts the input array to a new shape.

Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
with arrays of different shapes efficiently without creating multiple copies of arrays.
Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.

Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
`(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

For example::

   broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
                                           [ 1.,  2.,  3.]])

The dimension which you do not want to change can also be kept as `0` which means copy the original value.
So with `shape=(2,0)`, we will obtain the same result as in the above example.

)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(BroadcastToParam::__FIELDS__())
.set_attr_parser(ParamParser<BroadcastToParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<BroadcastToParam>)
.set_attr<FInferShape>("FInferShape", BroadcastToInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseSamePrecision)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(4);

// binary broadcast op
inline bool BinaryBroadcastShape(const cvm::NodeAttrs& attrs,
                                 std::vector<TShape>* in_attrs,
                                 std::vector<TShape>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 2U);
  VERIFY_EQ(out_attrs->size(), 1U);
  const TShape& lhs = (*in_attrs)[0];
  const TShape& rhs = (*in_attrs)[1];

  // avoid pre-mature shape inference.
  if (lhs.ndim() == 0 || rhs.ndim() == 0) return false;

  if (lhs == rhs) {
    CVM_ASSIGN_INPUT_SHAPE(attrs, *out_attrs, 0, lhs);
    return true;
  }
  TShape out(std::max(lhs.ndim(), rhs.ndim()));
  dim_t bl = out.ndim() - lhs.ndim();
  dim_t br = out.ndim() - rhs.ndim();
  for (dim_t i = 0; i < out.ndim(); ++i) {
    dim_t l = 1, r = 1;
    if (i >= bl) l = lhs[i - bl];
    if (i >= br) r = rhs[i - br];
    if (l != r) {
      if (l == 0 || r == 0) {
        out[i] = 0;
      } else {
        VERIFY(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes "
          << lhs << " " << rhs << ", l=" << l << ", r=" << r;
        out[i] = std::max(l, r);
      }
    } else {
      out[i] = l;
    }
  }
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out);
  return true;
}

inline bool BinaryBroadcastCorrectLayout(const NodeAttrs& attrs,
                                         std::vector<Layout> *ilayouts,
                                         const std::vector<Layout> *last_ilayouts,
                                         std::vector<Layout> *olayouts) {
  VERIFY_EQ(ilayouts->size(), 2U);
  VERIFY_EQ(olayouts->size(), 1U);
  Layout lhs = (*ilayouts)[0];
  Layout rhs = (*ilayouts)[1];
  Layout out(Layout::Undef());

  if (lhs.defined() && rhs.defined()) {
    if (lhs == rhs) {
      CVM_ASSIGN_LAYOUT(*olayouts, 0, lhs);
      return true;
    }
    // For example, NCHW <-> CHW, N16nCH16cW <-> HCW16c, etc, are broadcast-convertible
    // because as the definition, CHW can broadcast with NCHW.
    // For the second case, we can convert HCW16c to CH16cW then it can broadcast with N16nCH16cW.
    // But CNHW <-> CHW, NCHW16n <-> CHW are not,
    // because not matter how we adjust the layout of 'CHW',
    // we can never have an 'N' between 'C' and "HW".
    size_t l_start = 0, r_start = 0;
    size_t l = 0, r = 0;
    bool find_first_match = false;
    while (l < lhs.ndim() && r < rhs.ndim()) {
      if (!rhs.contains(Layout::to_superdim(lhs[l]))) {
        VERIFY(!find_first_match) << lhs << " and " << rhs << " are not broadcast-convertible";
        l_start = ++l;
      } else if (!lhs.contains(Layout::to_superdim(rhs[r]))) {
        VERIFY(!find_first_match) << lhs << " and " << rhs << " are not broadcast-convertible";
        r_start = ++r;
      } else {
        find_first_match = true;
        ++l; ++r;
      }
    }
    if (l_start > 0 && r_start > 0) {
      LOG(FATAL) << lhs << " and " << rhs << " are not broadcast-convertible";
    } else if (l_start > 0) {
      rhs = lhs.sublayout(l_start, lhs.ndim()-l_start);
      out = lhs;
    } else if (r_start > 0) {
      lhs = rhs.sublayout(r_start, rhs.ndim()-r_start);
      out = rhs;
    } else {
      // prior to keep left layout
      rhs = lhs;
      out = lhs;
    }
  } else if (lhs.defined()) {
    const Layout& last_lhs = last_ilayouts->at(0);
    if (last_lhs.defined()) {
      VERIFY(lhs.convertible(last_lhs)) << "current lhs layout " << lhs
                                       << " cannot be converted to the original one " << last_lhs;
      lhs = last_lhs;
      // cannot decide output layout
    }
  } else if (rhs.defined()) {
    const Layout& last_rhs = last_ilayouts->at(1);
    if (last_rhs.defined()) {
      VERIFY(rhs.convertible(last_rhs)) << "current rhs layout " << rhs
                                       << " cannot be converted to the original one " << last_rhs;
      rhs = last_rhs;
      // cannot decide output layout
    }
  }
  CVM_ASSIGN_LAYOUT(*ilayouts, 0, lhs);
  CVM_ASSIGN_LAYOUT(*ilayouts, 1, rhs);
  CVM_ASSIGN_LAYOUT(*olayouts, 0, out);
  return true;
}

#define CVM_REGISTER_BINARY_BROADCAST_OP(name, TOPIOp)             \
  CVM_REGISTER_OP(name)                                            \
  .set_num_inputs(2)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<FInferShape>("FInferShape", BinaryBroadcastShape)       \
  .set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)           \
  .set_attr<FCorrectLayout>("FCorrectLayout",                       \
    BinaryBroadcastCorrectLayout)                                   \
  .set_attr<FInplaceOption>("FInplaceOption",                       \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};     \
    })                                                              \
  .add_argument("lhs", "Tensor", "first input")                     \
  .add_argument("rhs", "Tensor", "second input")


CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_add, add)
.add_alias("__add_symbol__")
.describe(R"code(Returns element-wise sum of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_add(x, y) = [[ 1.,  1.,  1.],
                          [ 2.,  2.,  2.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePlusonePrecision);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_sub, subtract)
.add_alias("__sub_symbol__")
.describe(R"code(Returns element-wise difference of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_sub(x, y) = [[ 1.,  1.,  1.],
                          [ 0.,  0.,  0.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePlusonePrecision);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_mul, multiply)
.add_alias("__mul_symbol__")
.describe(R"code(Returns element-wise product of the input arrays with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_mul(x, y) = [[ 0.,  0.,  0.],
                          [ 1.,  1.,  1.]]
)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseSumPrecision);


CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_div, divide)
.add_alias("__div_symbol__")
.describe(R"code(Returns element-wise division of the input arrays with broadcasting.

Example::

   x = [[ 6.,  6.,  6.],
        [ 6.,  6.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_div(x, y) = [[ 3.,  3.,  3.],
                          [ 2.,  2.,  2.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseFirstPrecision);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_mod, mod)
.add_alias("__mod_symbol__")
.describe(R"code(Returns element-wise mod of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_mod(x, y) = [[ 1.,  0.,  1.],
                          [ 1.,  2.,  0.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseSecondPrecision);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_max, maximum)
.add_alias("__max_symbol__")
.describe(R"code(Returns element-wise max of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_max(x, y) = [[ 2.,  2.,  3.],
                          [ 4.,  5.,  6.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseMaxPrecision);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_min, minimum)
.add_alias("__min_symbol__")
.describe(R"code(Returns element-wise minimum of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_min(x, y) = [[ 1.,  2.,  2.],
                          [ 3.,  3.,  3.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseMaxPrecision);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_left_shift, left_shift)
.add_alias("__left_shift_symbol__")
.describe(R"code(Returns element-wise x << y of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 2.],
        [ 1.]]

   broadcast_left_shift(x, y) = [[ 4.,  8.,  12.],
                                 [ 8.,  10., 12.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<32>);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_right_shift, right_shift)
.add_alias("__right_shift_symbol__")
.describe(R"code(Returns element-wise x >> y of the input arrays with broadcasting.

Example::

   x = [[ 4.,  8.,  12.],
        [ 8.,  10., 12.]]

   y = [[ 2.],
        [ 1.]]

   broadcast_right_shift(x, y) = [[ 1.,  2.,  3.],
                                  [ 4.,  5.,  6.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<8>);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_greater, greater)
.add_alias("__greater_symbol__")
.describe(R"code(Returns element-wise x > y of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_greater(x, y) = [[ 0.,  0.,  1.],
                              [ 1.,  1.,  1.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_less, less)
.add_alias("__less_symbol__")
.describe(R"code(Returns element-wise x < y of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 2.],
        [ 3.]]

   broadcast_less(x, y) = [[ 1.,  0.,  0.],
                           [ 0.,  0.,  0.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_equal, equal)
.add_alias("__equal_symbol__")
.describe(R"code(Returns element-wise x == y of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 2.],
        [ 5.]]

   broadcast_equal(x, y) = [[ 0.,  1.,  0.],
                            [ 0.,  1.,  0.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_not_equal, not_equal)
.add_alias("__not_equal_symbol__")
.describe(R"code(Returns element-wise x != y of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 2.],
        [ 4.]]

   broadcast_not_equal(x, y) = [[ 1.,  0.,  1.],
                                [ 0.,  1.,  1.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_greater_equal, greater_equal)
.add_alias("__greater_equal_symbol__")
.describe(R"code(Returns element-wise x >= y of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 2.],
        [ 6.]]

   broadcast_greater_equal(x, y) = [[ 0.,  1.,  1.],
                                    [ 0.,  0.,  1.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>);

CVM_REGISTER_BINARY_BROADCAST_OP(broadcast_less_equal, less_equal)
.add_alias("__less_equal_symbol__")
.describe(R"code(Returns element-wise x <= y of the input arrays with broadcasting.

Example::

   x = [[ 1.,  2.,  3.],
        [ 4.,  5.,  6.]]

   y = [[ 1.],
        [ 5.]]

   broadcast_less_equal(x, y) = [[ 1.,  0.,  0.],
                                 [ 1.,  1.,  0.]]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwisePrecision<1>);

}  // namespace top
}  // namespace cvm
