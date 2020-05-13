/*!
 *  Copyright (c) 2017 by Contributors
 * \file cvm/top/tensor.h
 * \brief Auxiliary param for tensor primitive.
 */
#ifndef CVM_TOP_TENSOR_H_
#define CVM_TOP_TENSOR_H_

#include <utils/base.h>
#include <utils/parameter.h>
#include <cvm/tuple.h>

namespace cvm {
namespace top {

struct ConcatenateParam : public utils::Parameter<ConcatenateParam> {
  int axis;
  CVMUTIL_DECLARE_PARAMETER(ConcatenateParam) {
    CVMUTIL_DECLARE_FIELD(axis).set_default(1)
    .describe("the axis to be concated.");
  }
};

struct ExpandDimsParam : public utils::Parameter<ExpandDimsParam> {
  int axis;
  int num_newaxis;
  CVMUTIL_DECLARE_PARAMETER(ExpandDimsParam) {
    CVMUTIL_DECLARE_FIELD(axis)
    .describe("the axis to be expanded.");
    CVMUTIL_DECLARE_FIELD(num_newaxis).set_lower_bound(1).set_default(1)
    .describe("Number of new axis to be inserted.");
  }
};

struct RepeatParam : public utils::Parameter<RepeatParam> {
  int repeats;
  int axis;

  CVMUTIL_DECLARE_PARAMETER(RepeatParam) {
    CVMUTIL_DECLARE_FIELD(repeats)
      .describe("The number of repetitions for each element.");
    CVMUTIL_DECLARE_FIELD(axis).set_default(0)
        .describe(" The axis along which to repeat values.");
  }
};

struct TileParam : public utils::Parameter<TileParam> {
  TShape reps;

  CVMUTIL_DECLARE_PARAMETER(TileParam) {
    CVMUTIL_DECLARE_FIELD(reps).set_default(TShape{0})
      .describe("The number of times for repeating the tensor a."
                "Each dim sizeof reps must be a positive integer.");
  }
};

// struct PadParam : public utils::Parameter<PadParam> {
  // TShape pad_width;
  // int pad_value;

  // CVMUTIL_DECLARE_PARAMETER(PadParam) {
    // CVMUTIL_DECLARE_FIELD(pad_width).set_default(TShape{0, 0, 0, 0, 0, 1, 0, 1})
      // .describe("Widths of the padding regions applied to the edges of each axis."
                // "It is a tuple of integer padding widths for each axis of the format"
                // "(before_1, after_1, ... , before_N, after_N)."
                // "It should be of length 2*N wgere N is the number of dimensions of the array."
                // "This is equivalent to pad_width in numpy.pad, but flattened.");
    // CVMUTIL_DECLARE_FIELD(pad_value).set_default(0)
      // .describe("The value used for padding.");
  // }
// };

struct SplitParam : public utils::Parameter<SplitParam> {
  // numpy convention, only support indices, not support list.
  TShape indices_or_sections;
  int axis;
  // additional hint whether it is equal_split mode
  // deduced from indices_or_sections
  bool equal_split;

  CVMUTIL_DECLARE_PARAMETER(SplitParam) {
    CVMUTIL_DECLARE_FIELD(indices_or_sections).set_default(TShape{0})
        .describe("Number of outputs to be splitted");
    CVMUTIL_DECLARE_FIELD(axis).set_default(1)
        .describe("the axis to be splitted.");
  }
};


struct TakeParam : public utils::Parameter<TakeParam> {
  utils::optional<int> axis;

  CVMUTIL_DECLARE_PARAMETER(TakeParam) {
    CVMUTIL_DECLARE_FIELD(axis).set_default(utils::optional<int>())
        .describe("the axis over which to select values.");
  }
};

struct StridedSliceParam : public utils::Parameter<StridedSliceParam> {
  // numpy convention, only support indices, not support list.
  TShape begin;
  TShape end;
  TShape stride;

  CVMUTIL_DECLARE_PARAMETER(StridedSliceParam) {
    CVMUTIL_DECLARE_FIELD(begin).set_default(TShape{0})
        .describe("Indices for begin of slice");
    CVMUTIL_DECLARE_FIELD(end).set_default(TShape{1})
        .describe("Indices for end of the slice");
    CVMUTIL_DECLARE_FIELD(stride).set_default(TShape{})
        .describe("Stride values of the slice");
  }
};

enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
  kInt16 = 7,
  kUint16 = 8,
  kUint32 = 9,
  kUint64 = 10,
};

enum IndicatorRuleFlag {
  kGT0 = 0,
  kLT0 = 1,
  kMax = 2,
  kMin = 3,
};

#define CVMUTIL_DECLARE_DTYPE_FIELD(name)                              \
  CVMUTIL_DECLARE_FIELD(name)                                          \
  .add_enum("float16", kFloat16)                                    \
  .add_enum("float32", kFloat32)                                    \
  .add_enum("float64", kFloat64)                                    \
  .add_enum("uint8",  kUint8)                                       \
  .add_enum("uint16", kUint16)                                      \
  .add_enum("uint32", kUint32)                                      \
  .add_enum("uint64", kUint64)                                      \
  .add_enum("int8",  kInt8)                                         \
  .add_enum("int16", kInt16)                                        \
  .add_enum("int32", kInt32)                                        \
  .add_enum("int64", kInt64)

struct CastParam : public utils::Parameter<CastParam> {
  int dtype;
  CVMUTIL_DECLARE_PARAMETER(CastParam) {
    CVMUTIL_DECLARE_DTYPE_FIELD(dtype)
    .describe("Output data type.");
  }
};

// struct IndicatorParam : public utils::Parameter<IndicatorParam> {
//   TShape axis;
//   bool exclude;
//   CVMUTIL_DECLARE_PARAMETER(IndicatorParam) {
//     CVMUTIL_DECLARE_FIELD(axis).set_default(TShape())
//     .describe(R"code(The axis or axes along which to perform the indicator rule.
//
//         The default, `axis=()`, will compute over all elements into a
//         scalar array with shape `(1,)`.
//
//         If `axis` is int, rule is applied on a particular axis.
//
//         If `axis` is a tuple of ints, rule is applied on all the axes
//         specified in the tuple.
//
//         If `exclude` is true, rule will be applied on the axes that are
//         NOT in axis instead.)code");
//     CVMUTIL_DECLARE_FIELD(exclude).set_default(false)
//     .describe("Whether to apply rule on axis that are NOT in axis instead.");
//   }
// };

struct ReshapeParam : public utils::Parameter<ReshapeParam> {
  Tuple<int64_t> shape;

  CVMUTIL_DECLARE_PARAMETER(ReshapeParam) {
    CVMUTIL_DECLARE_FIELD(shape);
  }
};

struct SqueezeParam : public utils::Parameter<SqueezeParam> {
  TShape axis;

  CVMUTIL_DECLARE_PARAMETER(SqueezeParam) {
    CVMUTIL_DECLARE_FIELD(axis).set_default(TShape())
    .describe("The axis to squeeze in the input tensor.");
  }
};

struct ScalarParam : public utils::Parameter<ScalarParam> {
  int scalar;

  CVMUTIL_DECLARE_PARAMETER(ScalarParam) {
    CVMUTIL_DECLARE_FIELD(scalar);
  }
};

struct FillValueParam : public utils::Parameter<FillValueParam> {
  int fill_value;

  CVMUTIL_DECLARE_PARAMETER(FillValueParam) {
    CVMUTIL_DECLARE_FIELD(fill_value)
    .describe("Scalar value to be filled");
  }
};

struct TransposeParam : public utils::Parameter<TransposeParam> {
  TShape axes;

  CVMUTIL_DECLARE_PARAMETER(TransposeParam) {
    CVMUTIL_DECLARE_FIELD(axes).set_default(TShape())
    .describe("Target axis order. By default the axes will be inverted.");
  }
};

struct FlipParam : public utils::Parameter<FlipParam> {
  int axis;
  CVMUTIL_DECLARE_PARAMETER(FlipParam) {
    CVMUTIL_DECLARE_FIELD(axis).set_default(0)
    .describe("the axis to be reveresed.");
  }
};

struct BroadcastToParam : public utils::Parameter<BroadcastToParam> {
  TShape shape;

  CVMUTIL_DECLARE_PARAMETER(BroadcastToParam) {
    CVMUTIL_DECLARE_FIELD(shape).set_default(TShape())
      .describe("The shape of the desired array."
                " We can set the dim to zero if it's same as the original."
                " E.g `A = broadcast_to(B, shape=(10, 0, 0))` ");
  }
};

struct ReduceParam : public utils::Parameter<ReduceParam> {
  TShape axis;
  bool keepdims;
  bool exclude;
  int dtype;

  CVMUTIL_DECLARE_PARAMETER(ReduceParam) {
    CVMUTIL_DECLARE_FIELD(axis).set_default(TShape())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");

    CVMUTIL_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    CVMUTIL_DECLARE_FIELD(exclude).set_default(false)
      .describe("Whether to perform reduction on axis that are NOT in axis instead.");
    CVMUTIL_DECLARE_DTYPE_FIELD(dtype).set_default(kInt32)
      .describe("Target data type.");
  }
};

struct InitOpWithScalarParam : public utils::Parameter<InitOpWithScalarParam> {
  TShape shape;
  int dtype;
  int fill_value;

  CVMUTIL_DECLARE_PARAMETER(InitOpWithScalarParam) {
    CVMUTIL_DECLARE_FIELD(shape).set_default(TShape());
    CVMUTIL_DECLARE_DTYPE_FIELD(dtype).set_default(kInt32)
      .describe("Target data type.");
    CVMUTIL_DECLARE_FIELD(fill_value).describe("Scalar value to fill");
  }
};

struct InitOpParam : public utils::Parameter<InitOpParam> {
  TShape shape;
  int dtype;

  CVMUTIL_DECLARE_PARAMETER(InitOpParam) {
    CVMUTIL_DECLARE_FIELD(shape).set_default(TShape());
    CVMUTIL_DECLARE_DTYPE_FIELD(dtype).set_default(kInt32)
      .describe("Target data type.");
  }
};

struct ElementWiseReduceParam : public utils::Parameter<ElementWiseReduceParam> {
  int num_args;
  CVMUTIL_DECLARE_PARAMETER(ElementWiseReduceParam) {
    CVMUTIL_DECLARE_FIELD(num_args).set_lower_bound(1)
      .describe("Number of inputs to be reduced.");
  }
};

struct MatMulParam : public utils::Parameter<MatMulParam> {
  bool transpose_a;
  bool transpose_b;

  CVMUTIL_DECLARE_PARAMETER(MatMulParam) {
    CVMUTIL_DECLARE_FIELD(transpose_a)
      .describe("If true then transpose the first input before dot.")
      .set_default(false);
    CVMUTIL_DECLARE_FIELD(transpose_b)
      .describe("If true then transpose the second input before dot.")
      .set_default(false);
  }
};

struct ClipParam : public utils::Parameter<ClipParam> {
  int32_t a_min, a_max;
  CVMUTIL_DECLARE_PARAMETER(ClipParam) {
    CVMUTIL_DECLARE_FIELD(a_min)
      .describe("Minimum value such that value smaller then this will be clipped.");
    CVMUTIL_DECLARE_FIELD(a_max)
      .describe("Maximum value such that value larger then this will be clipped.");
  }
};

struct SliceLikeParam : public utils::Parameter<SliceLikeParam> {
  Tuple<int> axis;
  CVMUTIL_DECLARE_PARAMETER(SliceLikeParam) {
    CVMUTIL_DECLARE_FIELD(axis).set_default(Tuple<int>())
      .describe("List of axes on which input data will be sliced according to the "
                "corresponding size of the second input. By default will slice "
                "on all axes. Negative axes are supported.");
  }
};

}  // namespace top
}  // namespace cvm

#endif  // CVM_TOP_TENSOR_H_
