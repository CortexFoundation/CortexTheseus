/*!
 *  Copyright (c) 2017 by Contributors
 * \file reduce.cc
 * \brief reduce operator.
 */
#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/op_attr_types.h>
#include <cvm/top/tensor.h>
#include <numeric>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {


// reduce
CVMUTIL_REGISTER_PARAMETER(ReduceParam);

inline TShape GetReduceAxes(const uint32_t indim,
                            const TShape& axis,
                            bool exclude) {
  if (axis.ndim() == 0) {
    TShape r_axes(indim);
    std::iota(r_axes.begin(), r_axes.end(), 0);
    return r_axes;
  }

  VERIFY_LT(axis[axis.ndim() - 1], indim)
    << "Reduction axis " << axis[axis.ndim() - 1]
    << " exceeds input dimensions " << indim;

  TShape in_axis = axis;
  for (auto& i : in_axis) {
    i = i < 0 ? i + indim : i;
    VERIFY_GE(i, 0) << "axis out of bounds in reduce operator";
    VERIFY_LT(i, indim) << "axis out of bounds in reduce operator";
  }
  std::sort(in_axis.begin(), in_axis.end());
  if (!exclude) return in_axis;
  TShape r_axis(indim - in_axis.ndim());
  for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
    if (j < in_axis.ndim() && i == in_axis[j]) {
        ++j;
        continue;
    }
    r_axis[k++] = i;
  }
  return r_axis;
}

inline TShape ReduceShapeImpl(const TShape& ishape,
                              const TShape& axis,
                              bool keepdims,
                              bool exclude) {
  uint32_t indim = ishape.ndim();
  TShape r_axes = GetReduceAxes(indim, axis, exclude);
  if (!r_axes.ndim()) return ishape;
  if (r_axes.ndim() == indim)
    return TShape(keepdims ? indim : 1);

  VERIFY(r_axes.ndim() < indim);
  if (keepdims) {
    TShape oshape(ishape);
    for (unsigned i = 0, j = 0; i < indim; ++i) {
      if (j >= r_axes.ndim() || i != r_axes[j]) continue;
      oshape[i] = 1;
      ++j;
    }
    return oshape;
  }

  TShape oshape(indim - r_axes.ndim());
  for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
    if (j < r_axes.ndim() && i == r_axes[j]) {
      ++j;
      continue;
    }
    oshape[k++] = ishape[i];
  }
  return oshape;
}

inline bool ReduceShape(const cvm::NodeAttrs& attrs,
                        std::vector<TShape>* in_attrs,
                        std::vector<TShape>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 1U);
  VERIFY_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 0) return false;
  const ReduceParam& param = cvm::get<ReduceParam>(attrs.parsed);
  CVM_ASSIGN_OUTPUT_SHAPE(
      attrs, *out_attrs, 0,
      ReduceShapeImpl((*in_attrs)[0], param.axis,
                      param.keepdims, param.exclude));
  return true;
}

inline bool CollapseShape(const cvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_attrs,
                          std::vector<TShape>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 2U);
  VERIFY_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 1) return false;
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, (*in_attrs)[1]);
  return true;
}

template<typename PType>
inline void AxesParamParser(cvm::NodeAttrs* attrs) {
  PType param;
  param.Init(attrs->dict);
  std::sort(&param.axis[0], &param.axis[param.axis.ndim()]);
  attrs->parsed = std::move(param);
}

#define CVM_REGISTER_BASE_REDUCE_OP(op)                                 \
  CVM_REGISTER_OP(op)                                                   \
  .add_arguments(ReduceParam::__FIELDS__())                              \
  .set_attr_parser(AxesParamParser<ReduceParam>)                         \
  .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReduceParam>) \
  .set_num_outputs(1)

#define CVM_REGISTER_REDUCE_OP(op)                                     \
  CVM_REGISTER_BASE_REDUCE_OP(op)                                      \
  .add_argument("data", "Tensor", "The input")                          \
  .set_attr<FInferShape>("FInferShape", ReduceShape)                    \
  .set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)               \
  .set_attr<FCorrectLayout>("FCorrectLayout",                           \
    ElemwiseFixedLayoutUnknownOut<1, 1>)                                \
  .set_num_inputs(1)

CVM_REGISTER_REDUCE_OP(sum)
.describe(R"code(Computes the sum of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  sum(data, axis=1)
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision",
  [](const NodeAttrs& attrs,
   std::vector<TShape>* shapes,
   std::vector<int>* iattr,
   std::vector<int>* oattr) -> bool {
  auto& param = cvm::get<ReduceParam>(attrs.parsed);
  int ndim = param.axis.ndim();
  if (ndim == 0) ndim = 1;
  (*oattr)[0] = ndim * iattr->at(0);
  return true;
});

CVM_REGISTER_REDUCE_OP(max)
.describe(R"code(Computes the max of array elements over given axes.

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseSamePrecision);

CVM_REGISTER_REDUCE_OP(min)
.describe(R"code(Computes the min of array elements over given axes.

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseSamePrecision);

CVM_REGISTER_BASE_REDUCE_OP(collapse_sum)
.add_argument("data", "Tensor", "The input")
.add_argument("as", "Tensor", "The reference")
.set_attr<FInferShape>("FInferShape", CollapseShape)
.set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<2, 1>)
.set_num_inputs(2)
.describe(R"code(Reduces lhs to the shape of rhs via sum)code" CVM_ADD_FILELINE);

inline bool InferFixedType(const NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 1U);
  VERIFY_EQ(out_attrs->size(), 1U);
  const ReduceParam& param = cvm::get<ReduceParam>(attrs.parsed);
  CVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, param.dtype);
  return true;
}

CVM_REGISTER_BASE_REDUCE_OP(argmax)
.describe(R"code(Creates an operation that finds the indices of the maximum
values over a given axis.

)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "The input")
.set_attr<FInferShape>("FInferShape", ReduceShape)
.set_attr<FInferType>("FInferType", InferFixedType)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_num_inputs(1);

CVM_REGISTER_BASE_REDUCE_OP(argmin)
.describe(R"code(Creates an operation that finds the indices of the minimum
values over a given axis.

)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "The input")
.set_attr<FInferShape>("FInferShape", ReduceShape)
.set_attr<FInferType>("FInferType", InferFixedType)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_num_inputs(1);

CVM_REGISTER_REDUCE_OP(mean)
  .describe(R"code(Computes the mean of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  mean(data)
  [3.22]

  mean(data, axis=[1,2])
  [ 2.  3.16666667  4.5]

)code" CVM_ADD_FILELINE)
.set_attr<FInferPrecision>("FInferPrecision", ElemwiseSamePrecision);

CVM_REGISTER_REDUCE_OP(prod)
  .describe(R"code(Computes the products of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  mean(data, axis=1)
  [35562240]

  mean(data, axis=[1,2])
  [ 36  480  2058]

)code" CVM_ADD_FILELINE);


}  // namespace top
}  // namespace cvm
