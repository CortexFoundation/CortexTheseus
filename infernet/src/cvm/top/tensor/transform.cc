/*!
 *  Copyright (c) 2017 by Contributors
 * \file transform.cc
 * \brief Injective transformation of shape or type.
 */
#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/op_attr_types.h>
#include <cvm/top/tensor.h>
#include <cctype>
#include <sstream>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {

//repeat
CVMUTIL_REGISTER_PARAMETER(RepeatParam);

inline bool RepeatShape(const cvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_attrs,
                           std::vector<TShape>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 1U);
  VERIFY_EQ(out_attrs->size(), 1U);
  const TShape& shp = (*in_attrs)[0];
  const int ndim = static_cast<int>(shp.ndim());

  const RepeatParam& param = cvm::get<RepeatParam>(attrs.parsed);
  const int repeats = param.repeats;
  const int axis = param.axis;
  VERIFY_GT(param.repeats, 0) 
    << "operator " << attrs.name << " repeats:" << param.repeats
    << " must greater than 0";
  VerifyAttrRange(axis, "repeat.axis", -ndim, ndim);
  const int pivot = axis < 0 ? ndim + axis : axis;
  std::vector<int64_t> oshape;
  for (int i = 0; i < pivot; ++i) {
    oshape.emplace_back(shp[i]);
  }
  oshape.emplace_back(shp[pivot] * repeats);
  for (int i = pivot+1; i < ndim; ++i) {
    oshape.emplace_back(shp[i]);
  }
  TShape out_shape(oshape.begin(), oshape.end());
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out_shape);
  return true;
}

CVM_REGISTER_OP(repeat)
.describe(R"code(Repeat elements of an array `repeats` times along axis `axis`

- **data**: The input data to the operator.

)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Source input")
.add_arguments(RepeatParam::__FIELDS__())
.set_attr_parser(ParamParser<RepeatParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<RepeatParam>)
.set_attr<cvm::FInferShape>("FInferShape", RepeatShape)
.set_attr<cvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(1);

// tile
CVMUTIL_REGISTER_PARAMETER(TileParam);

inline bool TileShape(const cvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_attrs,
                           std::vector<TShape>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 1U);
  VERIFY_EQ(out_attrs->size(), 1U);
  const TShape& shp = (*in_attrs)[0];
  uint32_t sdim = shp.ndim();

  const TileParam& param = cvm::get<TileParam>(attrs.parsed);
  const auto& reps = param.reps;
  uint32_t rdim = reps.ndim();
  VERIFY(rdim > 0)
    << "repetition array is not defined. data.ndim = " << sdim;
  for (size_t i = 0; i < rdim; ++i) {
    VerifyAttrRange(reps[i], "tile.reps", 1);
  }

  uint32_t odim = std::max(sdim, rdim);
  std::vector<int64_t> oshape(odim);
  for (size_t i = 0; i < odim; ++i) {
    const auto s = i < sdim ? shp[sdim-1-i] : 1;
    const auto r = i < rdim ? reps[rdim-1-i] : 1;
    oshape[odim-1-i] = s * r;
  }
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0,
      TShape(oshape.begin(), oshape.end()));
  return true;
}

CVM_REGISTER_OP(tile)
.describe(R"code(Repeat the whole array multiple times.

- **data**: The input data to the operator.

)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Source input")
.add_arguments(TileParam::__FIELDS__())
.set_attr_parser(ParamParser<TileParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<TileParam>)
.set_attr<cvm::FInferShape>("FInferShape", TileShape)
.set_attr<cvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(1);



// flatten
inline bool FlattenInferShape(const NodeAttrs& attrs,
                              std::vector<TShape>* in_attrs,
                              std::vector<TShape>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 1U) << "Input: [data]";
  VERIFY_EQ(out_attrs->size(), 1U);
  const TShape &dshape = (*in_attrs)[0];
  uint32_t target_dim = 1;
  for (uint32_t i = 1; i < dshape.ndim(); ++i) {
    target_dim *= dshape[i];
  }
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0,
                           TShape({dshape[0], target_dim}));
  return true;
}

CVM_REGISTER_OP(flatten)
.describe(R"code(Flattens the input into a 2-D array.

For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
the input array into an output array of shape ``(d1, d2*...*dk)``.

Example::

    x = [[
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [   [1,2,3],
        [4,5,6],
        [7,8,9]
    ]],

    flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
       [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]

)code" CVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", FlattenInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.add_argument("data", "Tensor", "Input data.")
.set_support_level(1)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision);

// concatenate
CVMUTIL_REGISTER_PARAMETER(ConcatenateParam);

inline bool ConcatenateInferShape(const NodeAttrs& attrs,
                                  std::vector<TShape>* in_shape,
                                  std::vector<TShape>* out_shape) {
  const ConcatenateParam& param = cvm::get<ConcatenateParam>(attrs.parsed);
  TShape dshape;
  dim_t size = 0;
  bool has_zero = false;
  VERIFY(!in_shape->empty());
  int ndim = in_shape->at(0).ndim();
  VerifyAttrRange(param.axis, "concatenate.axis", -ndim, ndim);
  int axis = param.axis >= 0 ? param.axis : ndim + param.axis;
  for(size_t i = 0; i < in_shape->size(); ++i){
    VERIFY_EQ(in_shape->at(i).ndim(), ndim);
  }
  for (size_t i = 0; i < in_shape->size(); ++i) {
    TShape tmp = (*in_shape)[i];
    if (tmp.ndim()) {
      VERIFY_LT(static_cast<dim_t>(axis), tmp.ndim())
          << "concat dim " << axis << " out of range of input shape " << tmp;
      has_zero = tmp[axis] == 0 || has_zero;
      size += tmp[axis];
      tmp[axis] = 0;
      VERIFY_EQ(shape_assign(&dshape, tmp), true);
    }
  }

  TShape tmp = (*out_shape)[0];
  if (tmp.ndim()) {
    VERIFY_LT(static_cast<dim_t>(axis), tmp.ndim())
        << "concat dim " << axis << " out of range of input shape " << tmp;
    tmp[axis] = 0;
    VERIFY_EQ(shape_assign(&dshape, tmp), true);
  }

  for (size_t i = 0; i < in_shape->size(); ++i) {
    CVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, i, dshape);
  }

  if (!has_zero) dshape[axis] = size;
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, dshape);
  return dshape.Size() != 0;
}

inline bool ConcatenateCorrectLayout(const NodeAttrs& attrs,
                                     std::vector<Layout> *ilayouts,
                                     const std::vector<Layout> *last_ilayouts,
                                     std::vector<Layout> *olayouts) {
  const ConcatenateParam& param = cvm::get<ConcatenateParam>(attrs.parsed);
  VERIFY_EQ(ilayouts->size(), last_ilayouts->size());
  VERIFY_EQ(olayouts->size(), 1U);

  Layout layout;
  if (!ilayouts->at(0).defined()) {
    layout = last_ilayouts->at(0);
  } else if (param.axis >= static_cast<int>(ilayouts->at(0).ndim())) {
    VERIFY(last_ilayouts->at(0).defined())
      << "Current input layout " << ilayouts->at(0)
      << " is invalid but last input layout is not "
         "defined for the first input.";
    layout = last_ilayouts->at(0);
  } else if (last_ilayouts->at(0).defined()
             && ilayouts->at(0)[param.axis]
                != last_ilayouts->at(0)[param.axis]) {
    layout = last_ilayouts->at(0);
  } else {
    layout = ilayouts->at(0);
  }

  for (size_t i = 0; i < ilayouts->size(); ++i) {
    CVM_ASSIGN_LAYOUT(*ilayouts, i, layout);
  }
  CVM_ASSIGN_LAYOUT(*olayouts, 0, layout);
  return true;
}

CVM_REGISTER_OP(concatenate)
.describe(R"code(Joins input arrays along a given axis.

The dimensions of the input arrays should be the same except the axis along
which they will be concatenated.
The dimension of the output array along the concatenated axis will be equal
to the sum of the corresponding dimensions of the input arrays.

Example::

   x = [[1,1],[2,2]]
   y = [[3,3],[4,4],[5,5]]
   z = [[6,6], [7,7],[8,8]]

   concatenate(x,y,z,axis=0) = [[ 1.,  1.],
                               [ 2.,  2.],
                               [ 3.,  3.],
                               [ 4.,  4.],
                               [ 5.,  5.],
                               [ 6.,  6.],
                               [ 7.,  7.],
                               [ 8.,  8.]]

   Note that you cannot concat x,y,z along dimension 1 since dimension
   0 is not the same for all the input arrays.

   concatenate(y,z,axis=1) = [[ 3.,  3.,  6.,  6.],
                             [ 4.,  4.,  7.,  7.],
                             [ 5.,  5.,  8.,  8.]]

)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor-or-Tensor[]", "List of arrays to concatenate")
.add_arguments(ConcatenateParam::__FIELDS__())
.set_attr_parser(ParamParser<ConcatenateParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ConcatenateParam>)
.set_attr<FInferShape>("FInferShape", ConcatenateInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ConcatenateCorrectLayout)
.set_num_outputs(1)
.set_num_inputs(kVarg)
.set_attr<FInferPrecision>("FInferPrecision", MaxInPrecision)
.set_support_level(1);

// expand_dims
CVMUTIL_REGISTER_PARAMETER(ExpandDimsParam);

inline bool ExpandDimsInferShape(const NodeAttrs& attrs,
                                 std::vector<TShape>* in_shape,
                                 std::vector<TShape>* out_shape) {
  const ExpandDimsParam& param = cvm::get<ExpandDimsParam>(attrs.parsed);
  VERIFY_EQ(in_shape->size(), 1U);
  const TShape& dshape = in_shape->at(0);
  int ndim = static_cast<int>(dshape.ndim());
  VerifyAttrRange(param.axis, "expand_dims.axis", -ndim-1, ndim+1);
  VerifyAttrRange(param.num_newaxis, "expand_dims.num_newaxis");
  int axis = param.axis < 0 ? ndim + param.axis + 1 : param.axis;
  std::vector<dim_t> oshape;
  for (int i = 0; i < axis; ++i) {
    oshape.push_back(dshape[i]);
  }
  for (int i = 0; i < param.num_newaxis; ++i) {
    oshape.push_back(1);
  }
  for (int i = axis; i < ndim; ++i) {
    oshape.push_back(dshape[i]);
  }
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0,
                           TShape(oshape.begin(), oshape.end()));
  return true;
}

CVM_REGISTER_OP(expand_dims)
.describe(R"code(Inserts a new axis of size 1 into the array shape

For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1, num_newaxis=5)``
will return a new array with shape ``(2,1,1,1,1,1,3,4)``.

)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input tensor")
.add_arguments(ExpandDimsParam::__FIELDS__())
.set_attr_parser(ParamParser<ExpandDimsParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ExpandDimsParam>)
.set_attr<FInferShape>("FInferShape", ExpandDimsInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_support_level(1);

// reshape
CVMUTIL_REGISTER_PARAMETER(ReshapeParam);

inline bool ReshapeInferShape(const NodeAttrs& attrs,
                              std::vector<TShape>* in_attrs,
                              std::vector<TShape>* out_attrs) {
  const ReshapeParam& param = cvm::get<ReshapeParam>(attrs.parsed);
  VERIFY_GT(param.shape.ndim(), 0);
  VERIFY_EQ(in_attrs->size(), 1U) << "Input: [data]";
  VERIFY_EQ(out_attrs->size(), 1U);

  const TShape &dshape = (*in_attrs)[0];

  const Tuple<int64_t>& target_shape = param.shape;
  std::vector<int64_t> oshape;
  dim_t src_idx = 0;
  int infer_idx = -1;

  for (dim_t i = 0; i < target_shape.ndim(); ++i) {
    int64_t svalue = target_shape[i];
    // special flag handling for shape inference.
    if (svalue > 0) {
      oshape.push_back(svalue);
      ++src_idx;
    } else if (svalue == 0) {
      // keep same
      VERIFY_LT(src_idx, dshape.ndim());
      oshape.push_back(dshape[src_idx++]);
    } else if (svalue == -1) {
      // inference based on rest
      VERIFY_LT(infer_idx, 0)
          << "One and only one dim can be inferred";
      infer_idx = i;
      oshape.push_back(1);
      ++src_idx;
    } else if (svalue == -2) {
      // copy all remaining dims from source
      while (src_idx < dshape.ndim()) {
        oshape.push_back(dshape[src_idx++]);
      }
    } else if (svalue == -3) {
      // merge two dims from source
      VERIFY_LT(src_idx + 1, dshape.ndim());
      dim_t d1 = dshape[src_idx++];
      dim_t d2 = dshape[src_idx++];
      oshape.push_back(d1 * d2);
    } else if (svalue == -4) {
      // split the source dim s into two dims
      // read the left dim and then the right dim (either can be -1)
      VERIFY_LT(i + 2, target_shape.ndim());
      VERIFY_LT(src_idx, dshape.ndim());
      dim_t d0 = dshape[src_idx++];
      int d1 = target_shape[++i];
      int d2 = target_shape[++i];
      VERIFY(d1 != -1 || d2 != -1) << "Split dims cannot both be -1.";
      if (d1 == -1) d1 = d0 / d2;
      if (d2 == -1) d2 = d0 / d1;
      VERIFY_EQ(d1 * d2, static_cast<int>(d0)) <<
          "Split dims " << d1 << ", " << d2 << " do not divide original dim " << d0;
      oshape.push_back(d1);
      oshape.push_back(d2);
    }
  }

  if (infer_idx >= 0) {
    if (dshape.Size() > 0) {
      int new_size = 1;
      for (int x : oshape) {
        new_size *= x;
      }
      oshape[infer_idx] = dshape.Size() / new_size;
    } else {
      oshape[infer_idx] = 0;
    }
  }
  TShape out_shape(oshape.begin(), oshape.end());
  VERIFY_EQ(out_shape.Size(), dshape.Size())
      << "Target shape size is different to source. "
      << "Target: " << out_shape
      << "\nSource: " << dshape;
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out_shape);
  return true;
}

CVM_REGISTER_OP(reshape)
.describe(R"code(Reshapes the input array.

Given an array and a shape, this function returns a copy of the array in the new shape.
The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.

Example::

  reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]

To give user more convenience in without doing manual shape inference,
some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}.
The significance of each is explained below:

- ``0``  copy this dimension from the input to the output shape.

  Example::

  - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
  - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)

- ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
  keeping the size of the new array same as that of the input array.
  At most one dimension of shape can be -1.

  Example::

  - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
  - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
  - input shape = (2,3,4), shape=(-1,), output shape = (24,)

- ``-2`` copy all/remainder of the input dimensions to the output shape.

  Example::

  - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
  - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
  - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)

- ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.

  Example::

  - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
  - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
  - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
  - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)

- ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).

  Example::

  - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
  - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)

)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_arguments(ReshapeParam::__FIELDS__())
.set_attr_parser(ParamParser<ReshapeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReshapeParam>)
.set_attr<FInferShape>("FInferShape", ReshapeInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(3);

// squeeze
CVMUTIL_REGISTER_PARAMETER(SqueezeParam);

inline bool SqueezeShape(const cvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_attrs,
                           std::vector<TShape>* out_attrs) {
  const SqueezeParam& param = cvm::get<SqueezeParam>(attrs.parsed);
  VERIFY_EQ(in_attrs->size(), 1U);
  VERIFY_EQ(out_attrs->size(), 1U);
  const TShape& shp = (*in_attrs)[0];

  int ndim = shp.ndim();
  std::vector<int64_t> oshape;
  if (param.axis.ndim() == 0) {
    for (int i = 0; i < ndim; ++i) {
      if (shp[i] != 1) {
        oshape.emplace_back(shp[i]);
      }
    }
  } else {
    std::unordered_set<dim_t> axis_checker;
    for (size_t i = 0; i < param.axis.ndim(); ++i) {
      VerifyAttrRange(param.axis[i], "squeeze.axis", -ndim, ndim);
      int real_axis;
      if (param.axis[i] < 0) {
        real_axis = param.axis[i] + ndim;
      } else {
        real_axis = param.axis[i];
      }
      axis_checker.insert(real_axis);
    }
    for (int i = 0; i < ndim; ++i) {
      if (axis_checker.find(i) == axis_checker.end()) {
        oshape.emplace_back(shp[i]);
      } else {
        VERIFY_EQ(shp[i], 1) << "The squeezed axis must have shape 1!"
                            << "Want to squeeze " << i
                            << ", which has shape" << shp[i];
      }
    }
  }
  if (oshape.size() == 0) {
    // Handles the case where all axes are squeezed.
    oshape.push_back(1);
  }
  TShape out_shape(oshape.begin(), oshape.end());
  VERIFY_EQ(out_shape.Size(), shp.Size())
      << "Target shape size is different to source. "
      << "Target: " << out_shape
      << "\nSource: " << shp;
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out_shape);
  return true;
}

CVM_REGISTER_OP(squeeze)
.describe(R"code(Squeeze axises in the array.

Examples::

  x = [[[0], [1], [2]]]
  x.shape = (1, 3, 1)

  squeeze(x) = [0, 1, 2]

  squeeze(x, 0) = [[0], [1], [2]]

  squeeze(x, (0, 2)) = [0, 1, 2]

)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Source input")
.add_arguments(SqueezeParam::__FIELDS__())
.set_attr_parser(ParamParser<SqueezeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<SqueezeParam>)
.set_attr<cvm::FInferShape>("FInferShape", SqueezeShape)
.set_attr<cvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(1);

// transpose
CVMUTIL_REGISTER_PARAMETER(TransposeParam);

inline bool TransposeShape(const cvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_attrs,
                           std::vector<TShape>* out_attrs) {
  const TransposeParam& param = cvm::get<TransposeParam>(attrs.parsed);
  VERIFY_EQ(in_attrs->size(), 1U);
  VERIFY_EQ(out_attrs->size(), 1U);
  const TShape& shp = (*in_attrs)[0];
  int ndim = shp.ndim();

  TShape ret(ndim);
  if (param.axes.ndim() == 0) {
    for (int i = 0; i < ndim; ++i) {
      ret[i] = shp[ndim - 1 - i];
    }
  } else {
    VERIFY_EQ(shp.ndim(), param.axes.ndim());
    TShape axes(param.axes);
    for (int i = 0; i < ndim; ++i) {
      int64_t new_axis = axes[i];
      VerifyAttrRange(new_axis, "transpose.axis", -ndim, ndim);
      if (new_axis < 0) {
        new_axis += ndim;
        axes[i] = new_axis;
      }
      for (int j = 0; j < ndim; ++j) {
        if (i != j) {
          VERIFY(new_axis != axes[j]) << "repeated axis in transpose";
        }
      }
      ret[i] = shp[new_axis];
    }
  }
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, ret);
  return true;
}

inline bool TransposeCorrectLayout(const NodeAttrs& attrs,
                                   std::vector<Layout> *ilayouts,
                                   const std::vector<Layout> *last_ilayouts,
                                   std::vector<Layout> *olayouts) {
  const TransposeParam& param = cvm::get<TransposeParam>(attrs.parsed);
  VERIFY_EQ(ilayouts->size(), 1U);
  VERIFY_EQ(olayouts->size(), 1U);

  const Layout& input = last_ilayouts->at(0).defined()
                        ? last_ilayouts->at(0)
                        : ilayouts->at(0);

  CVM_ASSIGN_LAYOUT(*ilayouts, 0, input);

  if (input.defined()) {
    std::ostringstream new_layout;
    if (param.axes.ndim() == 0) {
      for (size_t i = 0; i < input.ndim(); ++i) {
        new_layout << input.at(input.ndim() - 1 - i);
      }
    } else {
      VERIFY_EQ(input.ndim(), param.axes.ndim());
      for (size_t i = 0; i < input.ndim(); ++i) {
        VERIFY(param.axes[i] < static_cast<int>(input.ndim()));
        new_layout << input.at(param.axes[i]);
      }
    }
    CVM_ASSIGN_LAYOUT(*olayouts, 0, Layout(new_layout.str()));
  }

  return true;
}

CVM_REGISTER_OP(transpose)
.describe(R"code(Permutes the dimensions of an array.

Examples::

  x = [[ 1, 2],
       [ 3, 4]]

  transpose(x) = [[ 1.,  3.],
                  [ 2.,  4.]]

  x = [[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]]]

  transpose(x) = [[[ 1.,  5.],
                   [ 3.,  7.]],

                  [[ 2.,  6.],
                   [ 4.,  8.]]]

  transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
                                 [ 5.,  6.]],

                                [[ 3.,  4.],
                                 [ 7.,  8.]]]
)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Source input")
.add_arguments(TransposeParam::__FIELDS__())
.set_attr_parser(ParamParser<TransposeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<TransposeParam>)
.set_attr<cvm::FInferShape>("FInferShape", TransposeShape)
.set_attr<cvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", TransposeCorrectLayout)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(4);

// strided_slice
CVMUTIL_REGISTER_PARAMETER(StridedSliceParam);

inline void StridedSliceParamParser(cvm::NodeAttrs* attrs) {
  StridedSliceParam param;
  param.Init(attrs->dict);
  attrs->parsed = std::move(param);
}

inline bool StridedSliceInferShape(const NodeAttrs& attrs,
                            std::vector<TShape>* in_shape,
                            std::vector<TShape>* out_shape) {
  const StridedSliceParam& param = cvm::get<StridedSliceParam>(attrs.parsed);
  const TShape& dshape = (*in_shape)[0];
  TShape oshape = dshape;
  dim_t num_axis = dshape.ndim();

  std::vector<int64_t> begin_vec;
  std::copy(param.begin.begin(), param.begin.end(), std::back_inserter(begin_vec));
  for (dim_t i = begin_vec.size(); i < num_axis; ++i) {
    begin_vec.push_back(0);
  }

  std::vector<int64_t> end_vec;
  std::copy(param.end.begin(), param.end.end(), std::back_inserter(end_vec));
  for (dim_t i = end_vec.size(); i < num_axis; ++i) {
    end_vec.push_back(dshape[i]);
  }

  std::vector<int64_t> stride_vec;
  std::copy(param.stride.begin(), param.stride.end(), std::back_inserter(stride_vec));
  for (dim_t i = stride_vec.size(); i < num_axis; ++i) {
    stride_vec.push_back(1);
  }

  for (dim_t i = 0; i < num_axis; ++i) {
    VERIFY(stride_vec[i] != 0);
    int64_t begin_range = stride_vec[i] < 0 ? -1 : 0;
    int64_t end_range = stride_vec[i] < 0 ? dshape[i] - 1 : dshape[i];
    int64_t begin = begin_vec[i] < 0 ? dshape[i] + begin_vec[i] : begin_vec[i];
    int64_t end = end_vec[i] < 0 ? dshape[i] + end_vec[i] : end_vec[i];
    begin = std::min(std::max(begin, begin_range), end_range);
    end = std::min(std::max(end, begin_range), end_range);

    int interval = std::abs(end - begin);
    int slice_size = static_cast<int>((interval
          + std::abs(stride_vec[i]) - 1) / std::abs(stride_vec[i]));
    VERIFY(stride_vec[i] < 0 ? (end < begin) : (begin < end))
      << ": Input [Begin=" << begin_vec[i] << ", End=" << end_vec[i]
      << "] is invalid for axis=" << i;
    oshape[i] = slice_size;
  }
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

CVM_REGISTER_OP(slice)
.describe(R"code(Strided slice of an array.

Examples::

  x = [[  1.,   4.,   7.,  10.],
       [  2.,   5.,   8.,  11.],
       [  3.,   6.,   9.,  12.]]

  strided_slice(x, begin=[0, 1], end=[2, 4], stride=[1, 1]) = [[ 4.,  7.,  10.],
                                                               [ 5.,  8.,  11.]]

  x = [[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]]]

  strided_slice(x, begin=[0, 0], end=[2, 2]) = [[[ 1.,  2.],
                                                 [ 3.,  4.]],

                                                [[ 5.,  6.],
                                                 [ 7.,  8.]]]
)code" CVM_ADD_FILELINE)
.add_alias("strided_slice")
.add_argument("data", "Tensor", "Array to be sliced")
.add_arguments(StridedSliceParam::__FIELDS__())
.set_attr_parser(StridedSliceParamParser)
.set_attr<FInferShape>("FInferShape", StridedSliceInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseArbitraryLayout<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_num_inputs(1)
.set_num_outputs(1)
.set_support_level(1);

// take
CVMUTIL_REGISTER_PARAMETER(TakeParam);

inline bool TakeInferShape(const NodeAttrs& attrs,
                           std::vector<TShape>* in_shape,
                           std::vector<TShape>* out_shape) {
  VERIFY_EQ(in_shape->size(), 2U);
  VERIFY_EQ(out_shape->size(), 1U);
  const TShape& dshape = (*in_shape)[0];
  const TShape& indicesshape = (*in_shape)[1];
  int ndim = dshape.ndim();

  const TakeParam& param = cvm::get<TakeParam>(attrs.parsed);
  TShape oshape((!param.axis ? 0: dshape.ndim() - 1) + indicesshape.ndim());
  if (!param.axis) {
    for (size_t j = 0; j < indicesshape.ndim(); ++j) {
      oshape[j] = indicesshape[j];
    }
  } else {
    int axis = param.axis.value();
    VerifyAttrRange(axis, "take.axis", -ndim, ndim);
    if (axis < 0) {
      axis += ndim;
    }
    
    size_t posi = 0;
    for (int i = 0; i < ndim; ++i) {
      if (i == axis) {
        for (size_t j = 0; j < indicesshape.ndim(); ++j) {
          oshape[posi++] = indicesshape[j];
        }
      } else {
        oshape[posi++] = dshape[i];
      }
    }
  }
  CVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 0, dshape);
  CVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, 1, indicesshape);
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return dshape.Size() != 0;
}

inline bool TakeInferType(const NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 2U);
  VERIFY_EQ(out_attrs->size(), 1U);
  VERIFY_EQ((*in_attrs)[1], kInt32);
  CVM_ASSIGN_INPUT_TYPE(attrs, *in_attrs, 0, (*in_attrs)[0]);
  CVM_ASSIGN_INPUT_TYPE(attrs, *in_attrs, 1, static_cast<int>(kInt32));
  CVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, (*in_attrs)[0]);
  return true;
}

inline bool TakeCorrectLayout(const NodeAttrs& attrs,
                              std::vector<Layout> *ilayouts,
                              const std::vector<Layout> *last_ilayouts,
                              std::vector<Layout> *olayouts) {
  VERIFY_EQ(ilayouts->size(), last_ilayouts->size());
  VERIFY_EQ(olayouts->size(), 1U);

  for (size_t i = 0; i < ilayouts->size(); ++i) {
    const Layout& input = last_ilayouts->at(i).defined() ?
                          last_ilayouts->at(i) : ilayouts->at(i);
    CVM_ASSIGN_LAYOUT(*ilayouts, i, input);
  }

  return true;
}

CVM_REGISTER_OP(take)
.describe(R"code(Take elements from an array along an axis.

When axis is not None, this function does the same thing as 'fancy' indexing
(indexing arrays using arrays); however, it can be easier to use if you need
elements along a given axis.

**Note** that when axis is none the flattened input array is used.

Examples::

  a = [[ 1, 2],
       [ 3, 4]]
  indices = [3, 0, 2]
  take(a, indices) = [ 4, 1, 3]

  a = [[ 1., 2.],
       [ 3., 4.]]
  indices = [1, 0]
  take(a, indices, axis=1) = [[ 2., 1.],
                              [ 4., 3.]]

  )code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Array to be indexed")
.add_argument("indices", "Tensor", "The indices of the values to extract")
.add_arguments(TakeParam::__FIELDS__())
.set_attr_parser(ParamParser<TakeParam>)
.set_attr<FInferShape>("FInferShape", TakeInferShape)
.set_attr<FInferType>("FInferType", TakeInferType)
.set_attr<FCorrectLayout>("FCorrectLayout", TakeCorrectLayout)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_num_inputs(2)
.set_num_outputs(1)
.set_support_level(3);

// cvm_lut
CVMUTIL_REGISTER_PARAMETER(CVMLUTParam);

inline bool LUTInferShape(const NodeAttrs& attrs,
						  std::vector<TShape>* in_shape,
						  std::vector<TShape>* out_shape) {
  VERIFY_EQ(in_shape->size(), 2U);
  VERIFY_EQ(out_shape->size(), 1U);
  const TShape& dshape = (*in_shape)[0];
  const TShape& lutshape = (*in_shape)[1];
  const CVMLUTParam &param = cvm::get<CVMLUTParam>(attrs.parsed);
  VERIFY_EQ(lutshape.Size(), param.in_dim);
  TShape oshape(dshape.ndim());
	for (size_t j = 0; j < dshape.ndim(); ++j) {
	  oshape[j] = dshape[j];
	}
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
	return true;
}

inline bool LUTInferType(const NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 2U);
  VERIFY_EQ(out_attrs->size(), 1U);
  CVM_ASSIGN_INPUT_TYPE(attrs, *in_attrs, 0, (*in_attrs)[0]);
  CVM_ASSIGN_INPUT_TYPE(attrs, *in_attrs, 1, (*in_attrs)[1]);
  CVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, (*in_attrs)[1]);
  return true;
}

inline bool LUTCorrectLayout(const NodeAttrs& attrs,
                              std::vector<Layout> *ilayouts,
                              const std::vector<Layout> *last_ilayouts,
                              std::vector<Layout> *olayouts) {
  VERIFY_EQ(ilayouts->size(), last_ilayouts->size());
  VERIFY_EQ(olayouts->size(), 1U);

  for (size_t i = 0; i < ilayouts->size(); ++i) {
    const Layout& input = last_ilayouts->at(i).defined() ?
                          last_ilayouts->at(i) : ilayouts->at(i);
    CVM_ASSIGN_LAYOUT(*ilayouts, i, input);
  }

  return true;
}

inline bool LUTInferPrecision(const NodeAttrs& attrs,
                                  std::vector<TShape>* shapes,
                                  std::vector<int>* iattr,
                                  std::vector<int>* oattr) {
  IN_PREC_CHECK(iattr, attrs.name);
  (*oattr)[0] = iattr->at(1);
  return true;
}

CVM_REGISTER_OP(cvm_lut)
.describe(R"doc(CVMLUT look up input with table.
)doc" CVM_ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CVMLUTParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<CVMLUTParam>)
.set_attr<FInferShape>("FInferShape", LUTInferShape)
.set_attr<FInferType>("FInferType", LUTInferType)
.set_attr<FInferPrecision>("FInferPrecision", LUTInferPrecision)
.add_argument("data", "Tensor", "input")
.add_argument("table", "Tensor", "The table to lookup")
.add_arguments(CVMLUTParam::__FIELDS__())
.set_support_level(4);


// SliceLike
CVMUTIL_REGISTER_PARAMETER(SliceLikeParam);

inline bool SliceLikeShape(const cvm::NodeAttrs& attrs,
                           std::vector<TShape>* in_attrs,
                           std::vector<TShape>* out_attrs) {
  VERIFY_EQ(in_attrs->size(), 2U);
  VERIFY_EQ(out_attrs->size(), 1U);
  const SliceLikeParam& param = cvm::get<SliceLikeParam>(attrs.parsed);
  const TShape& src_shape = in_attrs->at(0);
  const TShape& target_shape = in_attrs->at(1);
  Tuple<dim_t> end_idx;
  end_idx = Tuple<dim_t>(src_shape);
  if (param.axis.ndim() == 0) {
    for (size_t i = 0; i < src_shape.ndim(); ++i) {
      if (i < target_shape.ndim()) {
        end_idx[i] = target_shape[i];
        VERIFY_LE(end_idx[i], src_shape[i])
          << "End index of axis " << i << " exceeds input shape: "
          << end_idx[i] << " vs " << src_shape[i];
      }
    }
  } else {
    for (auto i : param.axis) {
      VerifyAttrRange(i, "slice_like.axis",
          -src_shape.ndim(), target_shape.ndim());
      if (i < 0) {
        i = src_shape.ndim() + i;
      }
      end_idx[i] = target_shape[i];
      VERIFY_LE(end_idx[i], src_shape[i])
        << "End index of axis " << i << " exceeds input shape: "
        << end_idx[i] << " vs " << src_shape[i];
    }
  }
  TShape out_shape = TShape(std::move(end_idx));
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, out_shape);
  return true;
}

CVM_REGISTER_OP(slice_like)
.describe(R"code(Slice the first input respect to the second input.
)code" CVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data to be sliced.")
.add_argument("slice_like", "Tensor", "Tensor with target shape")
.set_num_inputs(2)
.set_num_outputs(1)
.add_arguments(SliceLikeParam::__FIELDS__())
.set_attr_parser(ParamParser<SliceLikeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<SliceLikeParam>)
.set_attr<FInferShape>("FInferShape", SliceLikeShape)
.set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseBinaryKeepLeftLayout)
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "slice_like"};
})
.set_support_level(4);

}  // namespace top
}  // namespace cvm
