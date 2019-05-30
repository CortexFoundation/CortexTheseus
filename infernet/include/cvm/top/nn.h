/*!
 *  Copyright (c) 2017 by Contributors
 * \file cvm/top/nn.h
 * \brief Auxiliary param for tensor primitive.
 */
#ifndef CVM_TOP_NN_H_
#define CVM_TOP_NN_H_

#include <utils/base.h>
#include <utils/parameter.h>
#include <cvm/tuple.h>
#include <cvm/layout.h>
#include <string>
#include "tensor.h"

namespace cvm {
namespace top {

struct DenseParam : public utils::Parameter<DenseParam> {
  int units;
  bool use_bias;

  CVMUTIL_DECLARE_PARAMETER(DenseParam) {
    CVMUTIL_DECLARE_FIELD(units).set_lower_bound(1)
    .describe("Number of hidden units of the dense transformation.");
    CVMUTIL_DECLARE_FIELD(use_bias).set_default(true)
    .describe("Whether to use bias parameter");
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};

struct CVMLUTParam : public dmlc::Parameter<CVMLUTParam> {
	int in_dim;
  DMLC_DECLARE_PARAMETER(CVMLUTParam) {
    DMLC_DECLARE_FIELD(in_dim)
      .describe("In dimension indicates the inputs value range.");
  }
};


struct CVMClipParam : public utils::Parameter<CVMClipParam> {
	int precision;
	bool is_sign;
  CVMUTIL_DECLARE_PARAMETER(CVMClipParam) {
    CVMUTIL_DECLARE_FIELD(precision)
      .describe("Precision such that value out of range this will be clipped.");
    CVMUTIL_DECLARE_FIELD(is_sign).set_default(true)
      .describe("Clip range is sign int or unsigned int.");
  }
};

struct CVMLeftShiftParam : public utils::Parameter<CVMLeftShiftParam> {
	int precision;
	bool is_sign;
	int shift_bit;
  CVMUTIL_DECLARE_PARAMETER(CVMLeftShiftParam) {
    CVMUTIL_DECLARE_FIELD(precision)
      .describe("Precision such that value out of range this will be clipped.");
    CVMUTIL_DECLARE_FIELD(is_sign).set_default(true)
      .describe("Clip range is sign int or unsigned int.");
		CVMUTIL_DECLARE_FIELD(shift_bit)
			.describe("Left shift bit.");
  }
};

struct CVMRightShiftParam : public utils::Parameter<CVMRightShiftParam> {
	int precision;
	bool is_sign;
	int shift_bit;
  CVMUTIL_DECLARE_PARAMETER(CVMRightShiftParam) {
    CVMUTIL_DECLARE_FIELD(precision)
      .describe("Precision such that value out of range this will be clipped.");
    CVMUTIL_DECLARE_FIELD(is_sign).set_default(true)
      .describe("Clip range is sign int or unsigned int.");
		CVMUTIL_DECLARE_FIELD(shift_bit)
			.describe("Left shift bit.");
  }
};


//  // Shared by softmax and log_softmax
//  struct SoftmaxParam : public utils::Parameter<SoftmaxParam> {
//    int axis;
//  
//    CVMUTIL_DECLARE_PARAMETER(SoftmaxParam) {
//      CVMUTIL_DECLARE_FIELD(axis).set_default(-1)
//          .describe("The axis to sum over when computing softmax.");
//    }
//  };
//  
//  struct LeakyReLUParam : public utils::Parameter<LeakyReLUParam> {
//    double alpha;
//  
//    CVMUTIL_DECLARE_PARAMETER(LeakyReLUParam) {
//      CVMUTIL_DECLARE_FIELD(alpha).set_lower_bound(0.0).set_default(0.25)
//          .describe("slope coefficient for the negative half axis.");
//    }
//  };
//  
//  struct PReLUParam : public utils::Parameter<PReLUParam> {
//    int axis;
//    CVMUTIL_DECLARE_PARAMETER(PReLUParam) {
//      CVMUTIL_DECLARE_FIELD(axis).set_default(1)
//        .describe("Specify which shape axis the channel is specified.");
//    }
//  };
//  
//  struct PadParam : public utils::Parameter<PadParam> {
//    float pad_value;
//    Tuple<Tuple<int> > pad_width;
//  
//    CVMUTIL_DECLARE_PARAMETER(PadParam) {
//      CVMUTIL_DECLARE_FIELD(pad_value).set_default(0.0)
//        .describe("The value to be padded.");
//      CVMUTIL_DECLARE_FIELD(pad_width)
//        .describe("Number of values padded to the edges of each axis, "
//                  "in the format of ((before_1, after_1), ... (before_N, after_N))");
//    }
//  };

struct Conv2DParam : public utils::Parameter<Conv2DParam> {
  int channels;
  TShape kernel_size;
  TShape strides;
  TShape padding;
  TShape dilation;
  int groups;
  std::string layout;
  std::string kernel_layout;
  std::string out_layout;
  int out_dtype;
  bool use_bias;

  CVMUTIL_DECLARE_PARAMETER(Conv2DParam) {
    CVMUTIL_DECLARE_FIELD(channels)
      .describe("The dimensionality of the output space"
                "i.e. the number of output channels in the convolution.");
    CVMUTIL_DECLARE_FIELD(kernel_size)
      .describe("Specifies the dimensions of the convolution window.");
    CVMUTIL_DECLARE_FIELD(strides).set_default(TShape({1, 1}))
      .describe("Specifies the strides of the convolution.");
    CVMUTIL_DECLARE_FIELD(padding).set_default(TShape({0, 0}))
      .describe("If padding is non-zero, then the input is implicitly zero-padded"
                "on both sides for padding number of points");
    CVMUTIL_DECLARE_FIELD(dilation).set_default(TShape({1, 1}))
      .describe("Specifies the dilation rate to use for dilated convolution.");
    CVMUTIL_DECLARE_FIELD(groups).set_default(1)
      .describe("Controls the connections between inputs and outputs."
                "At groups=1, all inputs are convolved to all outputs."
                "At groups=2, the operation becomes equivalent to having two convolution"
                "layers side by side, each seeing half the input channels, and producing"
                "half the output channels, and both subsequently concatenated.");
    CVMUTIL_DECLARE_FIELD(layout).set_default("NCHW")
      .describe("Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
    CVMUTIL_DECLARE_FIELD(out_layout).set_default("__undef__")
      .describe("Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Default to be same as input layout.");
    CVMUTIL_DECLARE_FIELD(kernel_layout).set_default("OIHW")
      .describe("Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
                "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
                "dimensions respectively.");
    CVMUTIL_DECLARE_DTYPE_FIELD(out_dtype)
      .add_enum("same", -1)
      .set_default(-1)
      .describe("Output data type, set to explicit type under mixed precision setting");

    CVMUTIL_DECLARE_FIELD(use_bias).set_default(true)
      .describe("Whether the layer uses a bias vector.");
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};

struct Conv2DTransposeParam : public utils::Parameter<Conv2DTransposeParam> {
  int channels;
  TShape kernel_size;
  TShape strides;
  TShape padding;
  TShape output_padding;
  TShape dilation;
  int groups;
  std::string layout;
  std::string kernel_layout;
  int out_dtype;
  bool use_bias;

  CVMUTIL_DECLARE_PARAMETER(Conv2DTransposeParam) {
    CVMUTIL_DECLARE_FIELD(channels)
      .describe("The dimensionality of the output space"
                "i.e. the number of output channels in the convolution.");
    CVMUTIL_DECLARE_FIELD(kernel_size)
      .describe("Specifies the dimensions of the convolution window.");
    CVMUTIL_DECLARE_FIELD(strides).set_default(TShape({1, 1}))
      .describe("Specifies the strides of the convolution.");
    CVMUTIL_DECLARE_FIELD(output_padding).set_default(TShape({0, 0}))
      .describe("Zero-padding added to one side of the output.");
    CVMUTIL_DECLARE_FIELD(padding).set_default(TShape({0, 0}))
      .describe("If padding is non-zero, then the input is implicitly zero-padded"
                "on both sides for padding number of points");
    CVMUTIL_DECLARE_FIELD(dilation).set_default(TShape({1, 1}))
      .describe("Specifies the dilation rate to use for dilated convolution.");
    CVMUTIL_DECLARE_FIELD(groups).set_default(1)
      .describe("Controls the connections between inputs and outputs."
                "At groups=1, all inputs are convolved to all outputs."
                "At groups=2, the operation becomes equivalent to having two convolution"
                "layers side by side, each seeing half the input channels, and producing"
                "half the output channels, and both subsequently concatenated.");
    CVMUTIL_DECLARE_FIELD(layout).set_default("NCHW")
      .describe("Dimension ordering of data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
    CVMUTIL_DECLARE_FIELD(kernel_layout).set_default("OIHW")
      .describe("Dimension ordering of data and weight. Can be 'OIHW', 'OIHW16o16i', etc."
                "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
                "dimensions respectively.");
    CVMUTIL_DECLARE_DTYPE_FIELD(out_dtype)
        .add_enum("same", -1)
        .set_default(-1)
        .describe("Output data type, set to explicit type under mixed precision setting");
    CVMUTIL_DECLARE_FIELD(use_bias).set_default(true)
      .describe("Whether the layer uses a bias vector.");
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};


struct MaxPool2DParam : public utils::Parameter<MaxPool2DParam> {
  TShape pool_size;
  TShape strides;
  TShape padding;
  std::string layout;
  bool ceil_mode;

  CVMUTIL_DECLARE_PARAMETER(MaxPool2DParam) {
    CVMUTIL_DECLARE_FIELD(pool_size)
      .describe("Size of the pooling windows..");
    CVMUTIL_DECLARE_FIELD(strides).set_default(TShape({1, 1}))
      .describe("Specifies the strides of the convolution.");
    CVMUTIL_DECLARE_FIELD(padding).set_default(TShape({0, 0}))
      .describe("If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on all sides"
                "two int : bottom, right will use same padding as top, left"
                "four int : padding width in the order of (top, left, bottom, right)");
    CVMUTIL_DECLARE_FIELD(layout).set_default("NCHW")
      .describe("Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
    CVMUTIL_DECLARE_FIELD(ceil_mode).set_default(false)
      .describe("When true, will use ceil instead of floor to compute the output shape.");
  }
};

/*
struct AvgPool2DParam : public utils::Parameter<AvgPool2DParam> {
  TShape pool_size;
  TShape strides;
  TShape padding;
  std::string layout;
  bool ceil_mode;
  bool count_include_pad;

  CVMUTIL_DECLARE_PARAMETER(AvgPool2DParam) {
    CVMUTIL_DECLARE_FIELD(pool_size)
      .describe("Size of the pooling windows..");
    CVMUTIL_DECLARE_FIELD(strides).set_default(TShape({1, 1}))
      .describe("Specifies the strides of the convolution.");
    CVMUTIL_DECLARE_FIELD(padding).set_default(TShape({0, 0}))
      .describe("If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on all sides"
                "two int : bottom, right will use same padding as top, left"
                "four int : padding width in the order of (top, left, bottom, right)");
    CVMUTIL_DECLARE_FIELD(layout).set_default("NCHW")
      .describe("Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
    CVMUTIL_DECLARE_FIELD(ceil_mode).set_default(false)
      .describe("When true, will use ceil instead of floor to compute the output shape.");
    CVMUTIL_DECLARE_FIELD(count_include_pad).set_default(false)
      .describe("When true, will include padding to compute the average");
  }
};
*/
struct GlobalPool2DParam : public utils::Parameter<GlobalPool2DParam> {
  std::string layout;

  CVMUTIL_DECLARE_PARAMETER(GlobalPool2DParam) {
    CVMUTIL_DECLARE_FIELD(layout).set_default("NCHW")
      .describe("Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
  }
};

struct LayoutTransformParam : public utils::Parameter<LayoutTransformParam> {
  std::string src_layout;
  std::string dst_layout;

  CVMUTIL_DECLARE_PARAMETER(LayoutTransformParam) {
    CVMUTIL_DECLARE_FIELD(src_layout).set_default("__undef__")
    .describe("Dimension ordering of data");
    CVMUTIL_DECLARE_FIELD(dst_layout).set_default("__undef__")
    .describe("Dimension ordering of data.");
  }
};

struct NonMaximumSuppressionParam : public dmlc::Parameter<NonMaximumSuppressionParam> {
  bool return_indices;
  float iou_threshold;
  bool force_suppress;
  int top_k;
  int id_index;
  int coord_start;
  int score_index;
  int max_output_size;
  bool invalid_to_bottom;
  DMLC_DECLARE_PARAMETER(NonMaximumSuppressionParam) {
    DMLC_DECLARE_FIELD(max_output_size).set_default(-1)
      .describe("Max number of output valid boxes for each instance."
                "By default all valid boxes are returned.");
    DMLC_DECLARE_FIELD(iou_threshold).set_default(0.5)
      .describe("Non-maximum suppression threshold.");
    DMLC_DECLARE_FIELD(force_suppress).set_default(false)
      .describe("Suppress all detections regardless of class_id.");
    DMLC_DECLARE_FIELD(top_k).set_default(-1)
      .describe("Keep maximum top k detections before nms, -1 for no limit.");
    DMLC_DECLARE_FIELD(coord_start).set_default(2)
      .describe("Start index of the consecutive 4 coordinates.");
    DMLC_DECLARE_FIELD(score_index).set_default(1)
      .describe("Index of the scores/confidence of boxes.");
    DMLC_DECLARE_FIELD(id_index).set_default(0)
      .describe("Axis index of id.");
    DMLC_DECLARE_FIELD(return_indices).set_default(true)
      .describe("Whether to return box indices in input data.");
    DMLC_DECLARE_FIELD(invalid_to_bottom).set_default(false)
      .describe("Whether to move all invalid bounding boxes to the bottom.");
  }
};

/*
struct MultiBoxPriorParam : public utils::Parameter<MultiBoxPriorParam> {
  Tuple<float> sizes;
  Tuple<float> ratios;
  Tuple<float> steps;
  Tuple<float> offsets;
  bool clip;

  CVMUTIL_DECLARE_PARAMETER(MultiBoxPriorParam) {
    CVMUTIL_DECLARE_FIELD(sizes).set_default(Tuple<float>({1.0}))
      .describe("List of sizes of generated MultiBoxPriores.");
    CVMUTIL_DECLARE_FIELD(ratios).set_default(Tuple<float>({1.0}))
    .describe("List of aspect ratios of generated MultiBoxPriores.");
    CVMUTIL_DECLARE_FIELD(steps).set_default(Tuple<float>({-1.0, -1.0}))
    .describe("Priorbox step across y and x, -1 for auto calculation.");
    CVMUTIL_DECLARE_FIELD(offsets).set_default(Tuple<float>({0.5, 0.5}))
    .describe("Priorbox center offsets, y and x respectively.");
    CVMUTIL_DECLARE_FIELD(clip).set_default(false)
    .describe("Whether to clip out-of-boundary boxes.");
  }
};

struct MultiBoxTransformLocParam : public utils::Parameter<MultiBoxTransformLocParam> {
  bool clip;
  float threshold;
  Tuple<float> variances;
  CVMUTIL_DECLARE_PARAMETER(MultiBoxTransformLocParam) {
    CVMUTIL_DECLARE_FIELD(clip).set_default(true)
      .describe("Clip out-of-boundary boxes.");
    CVMUTIL_DECLARE_FIELD(threshold).set_default(0.01)
    .describe("Threshold to be a positive prediction.");
    CVMUTIL_DECLARE_FIELD(variances).set_default(Tuple<float>({0.1f, 0.1f, 0.2f, 0.2f}))
    .describe("Variances to be decoded from box regression output.");
  }
};

struct LRNParam : public utils::Parameter<LRNParam> {
  int size;
  int axis;
  float alpha;
  float beta;
  float bias;

  CVMUTIL_DECLARE_PARAMETER(LRNParam) {
    CVMUTIL_DECLARE_FIELD(size)
      .describe("The size of the local region to be considered for normalization.");
    CVMUTIL_DECLARE_FIELD(axis)
      .describe("input data layout channel axis");
    CVMUTIL_DECLARE_FIELD(alpha)
      .describe("The scaling parameter.");
    CVMUTIL_DECLARE_FIELD(beta)
      .describe("The exponent parameter.");
    CVMUTIL_DECLARE_FIELD(bias)
      .describe("The offset parameter.");
  }
  // constants
  static const constexpr int kData = 0;
};

*/
}  // namespace top
}  // namespace cvm

#endif  // CVM_TOP_NN_H_
