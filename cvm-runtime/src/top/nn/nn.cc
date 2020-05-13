/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */
#include <cvm/op.h>
#include <cvm/node.h>
#include <cvm/layout.h>
#include <cvm/op_attr_types.h>
#include <cvm/dlpack.h>
#include <cvm/top/nn.h>
#include "nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace cvm {
namespace top {

// dense
CVMUTIL_REGISTER_PARAMETER(DenseParam);

inline bool DenseInferPrecision(const NodeAttrs& attrs,
                                std::vector<TShape>* shapes,
                                std::vector<int>* iattr,
                                std::vector<int> *oattr){
  VERIFY_LE(iattr->at(0), 8)
    << "Dense " << attrs.name
    << " input must be INT8 vs. INT" << iattr->at(0);
  VERIFY_LE(iattr->at(1), 8)
    << "Dense " << attrs.name
    << " weight must be INT8 vs. INT" << iattr->at(1);

  int64_t max_size = shapes->at(0)[1];
  int oprec = iattr->at(0) + iattr->at(1);
  oprec += GetReduceSumBit(max_size);

  auto& param = cvm::get<cvm::top::DenseParam>(attrs.parsed);
  if (param.use_bias) {
    int bias_prec = iattr->at(2);
    oprec = std::max(oprec, bias_prec) + 1;
  }
  (*oattr)[0] = oprec;
  return true;
}

inline bool DenseInferShape(const cvm::NodeAttrs& attrs,
                            std::vector<TShape>* in_shape,
                            std::vector<TShape>* out_shape) {
  const DenseParam& param = cvm::get<DenseParam>(attrs.parsed);
  if (param.use_bias) {
    VERIFY_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    VERIFY_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  VERIFY_EQ((*in_shape)[0].ndim(), 2U) << "dense require 2-D data";
  VERIFY_EQ((*in_shape)[1].ndim(), 2U) << "dense require 2-D weight";
  VERIFY_EQ(out_shape->size(), 1U);
  // reverse infer
  if ((*out_shape)[0].ndim() != 0) {
    TShape dshape = (*out_shape)[0];
    dshape[dshape.ndim() - 1] = 0;
    CVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, DenseParam::kData, dshape);
  }
  dim_t num_inputs = 0;
  if ((*in_shape)[DenseParam::kData].ndim() != 0) {
    TShape oshape = (*in_shape)[DenseParam::kData];
    num_inputs = oshape[oshape.ndim() - 1];
    oshape[oshape.ndim() - 1] = param.units;
    CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  }
  CVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, DenseParam::kWeight,
                          TShape({param.units, num_inputs}));
  if (param.use_bias) {
    CVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, DenseParam::kBias, TShape({param.units}));
  }
  return true;
}

CVM_REGISTER_OP(dense)
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T + b`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **bias**: `(units,)`
- **out**: `(x1, x2, ..., xn, units)`

The learnable parameters include both ``weight`` and ``bias``.

If ``use_bias`` is set to be false, then the ``bias`` term is ignored.

)code" CVM_ADD_FILELINE)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(DenseParam::__FIELDS__())
.set_attr_parser(ParamParser<DenseParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DenseParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<DenseParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<DenseParam>)
.set_attr<FInferShape>("FInferShape", DenseInferShape)
.set_attr<FInferType>("FInferType", ElemwiseType<-1, 1>)
// leave weight & bias layout undefined
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutCopyToOut<1, 1>)
.set_attr<FInferPrecision>("FInferPrecision", DenseInferPrecision)
.set_support_level(1);

// relu
CVM_REGISTER_ELEMWISE_UNARY_OP(relu)
.describe(R"code(Computes rectified linear.

.. math::
   max(input, 0)

)code" CVM_ADD_FILELINE)
.add_alias("nn.relu")
.set_attr<FInferPrecision>("FInferPrecision", SamePrecision)
.set_support_level(1);

}  // namespace top
}  // namespace cvm
