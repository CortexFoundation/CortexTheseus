/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise_op_common.h
 * \brief Common operator utilities
 */
#ifndef CVM_TOP_ELEMWISE_OP_COMMON_H_
#define CVM_TOP_ELEMWISE_OP_COMMON_H_

#include <cvm/layout.h>
#include <cvm/top/nn.h>
#include <string>
#include <vector>
#include <utility>
#include <functional>
#include "op_common.h"

#define CORTEX_LOG2(x) (32 - __builtin_clz(unsigned(x)))

namespace cvm {
namespace top {

template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         std::string (*attr_string)(const AttrType&),
         int n_in = -1, int n_out = -1>
inline bool ElemwiseAttr(const cvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs,
                         const AttrType& none) {
  AttrType dattr = none;
  size_t in_size = in_attrs->size();
  size_t out_size = out_attrs->size();
  if (n_in != -1)
    in_size = static_cast<size_t>(n_in);
  if (n_out != -1)
    out_size = static_cast<size_t>(n_out);

  auto deduce = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        VERIFY(assign(&dattr, (*vec)[i]))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string((*vec)[i]);
      }
    };
  deduce(in_attrs, in_size, "input");
  if (reverse_infer) deduce(out_attrs, out_size, "output");

  auto write = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        VERIFY(assign(&(*vec)[i], dattr))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string((*vec)[i]);
      }
    };
  write(in_attrs, in_size, "input");
  write(out_attrs, out_size, "output");

  if (is_none(dattr)) return false;
  return true;
}

template<int n_in, int n_out>
inline bool ElemwiseShape(const NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  if (n_in != -1) {
    VERIFY_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    VERIFY_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<TShape, shape_is_none, shape_assign, true, shape_string>(
    attrs, in_attrs, out_attrs, TShape());
}

template<int def_v>
inline bool ElemwisePrecision(const NodeAttrs& attrs,
                                  std::vector<TShape>* shapes,
																	std::vector<int>* iattr,
																	std::vector<int>* oattr) {
  for (int& v : *oattr) {
    v = def_v;
  }
  return true;
}

inline bool ElemwiseSamePrecision(const NodeAttrs& attrs,
                                  std::vector<TShape>* shapes,
																	std::vector<int>* iattr,
																	std::vector<int>* oattr) {
  int def_v = -1;
  for (int v : *iattr) {
    if (v != -1) {
      def_v = v; break;
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    if (v == -1) v = def_v;
  }
  return true;
}

inline bool ElemwisePlusonePrecision(const NodeAttrs& attrs,
		                                 std::vector<TShape>* shapes,
																		 std::vector<int>* iattr,
																		 std::vector<int>* oattr) {
  int def_v = -1;
  for (int v : *iattr) {
    if (v > def_v) {
      def_v = v;
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v + 1;
  }
  for (int& v : *iattr) {
    if (v == -1) v = def_v;
  }
  return true;
}

inline bool ElemwiseMaxPrecision(const NodeAttrs& attrs,
		                                 std::vector<TShape>* shapes,
																		 std::vector<int>* iattr,
																		 std::vector<int>* oattr) {
  int def_v = -1;
  for (int v : *iattr) {
    if (v > def_v) {
      def_v = v;
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    if (v == -1) v = def_v;
  }
  return true;
}

inline bool ElemwiseSumPrecision(const NodeAttrs& attrs,
                                 std::vector<TShape>* shapes,
																 std::vector<int>* iattr,
																 std::vector<int>* oattr) {
  int def_v = 0;
  for (int v : *iattr) {
    if (v == -1) {
      return false;
    }
    def_v += v;
  }
  for (int& v : *oattr) {
    v = def_v;
  }
  return true;
}

inline bool ElemwiseSecondPrecision(const NodeAttrs& attrs,
		                               std::vector<TShape>* shapes,
																	 std::vector<int>* iattr,
																	 std::vector<int>* oattr) {
  if (iattr->size() < 2) return false;
	int def_v = iattr->at(1);
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    if (v == -1) v = def_v;
  }
  return true;
}

inline bool ElemwiseFirstPrecision(const NodeAttrs& attrs,
                                   std::vector<TShape>* shapes,
                                   std::vector<int>* iattr,
                                   std::vector<int>* oattr)
{
  if (iattr->size() == 0) return false;
  int def_v = iattr->at(0);
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    if (v == -1) v = def_v;
  }
  return true;
}

template<int n_in, int n_out>
inline bool ElemwiseType(const NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  if (n_in != -1) {
    VERIFY_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    VERIFY_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

inline bool ElementWiseReduceShape(const NodeAttrs& attrs,
                                   std::vector<TShape> *in_attrs,
                                   std::vector<TShape> *out_attrs) {
  VERIFY_EQ(out_attrs->size(), 1);
  return ElemwiseAttr<TShape, shape_is_none, shape_assign, true, shape_string>(
    attrs, in_attrs, out_attrs, TShape());
}

inline bool ElementWiseReduceType(const NodeAttrs& attrs,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  VERIFY_EQ(out_attrs->size(), 1);
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

template<int n_in, int n_out>
inline bool ElemwiseFixedLayout(const NodeAttrs& attrs,
                                std::vector<Layout> *in_layouts,
                                const std::vector<Layout> *last_in_layouts,
                                std::vector<Layout> *out_layouts,
                                const std::function<Layout(const Layout& in)>& finfer) {
  const size_t in_size = (n_in == -1) ? in_layouts->size() : static_cast<size_t>(n_in);
  const size_t out_size = (n_out == -1) ? out_layouts->size() : static_cast<size_t>(n_out);

  auto deduce = [&](Layout *target, const std::vector<Layout> *vec,
                    size_t size, const char *name) {
    for (size_t i = 0; i < size; ++i) {
      if (vec->at(i).defined()) {
        if (!target->defined()) {
          *target = vec->at(i);
        }
        VERIFY_EQ(*target, vec->at(i))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << *target
          << ", got " << vec->at(i);
      }
    }
  };

  Layout in, last_in, out;
  deduce(&in, in_layouts, in_size, "input");
  deduce(&last_in, last_in_layouts, in_size, "input (last infer pass)");
  deduce(&out, out_layouts, out_size, "output");

  if (!last_in.defined()) {
    last_in = in;
  } else {
    // else we copy in_layout produced by last infer pass to in_layout,
    // and let LayoutTransform pass
    // to insert an layout_transform node to fix the input layout.
    in = last_in;
  }

  out = finfer(in);

  auto write = [](std::vector<Layout> *vec, Layout& value, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      vec->at(i) = value;
    }
  };
  if (in.defined()) write(in_layouts, in, in_size);
  if (out.defined()) write(out_layouts, out, out_size);

  return true;
}

/*! \brief Fix the input layout as the previous inferred (if any) and copy to output */
template<int n_in, int n_out>
inline bool ElemwiseFixedLayoutCopyToOut(const NodeAttrs& attrs,
                                         std::vector<Layout> *in_layouts,
                                         const std::vector<Layout> *last_in_layouts,
                                         std::vector<Layout> *out_layouts) {
  return ElemwiseFixedLayout<n_in, n_out>(
    attrs, in_layouts, last_in_layouts, out_layouts, [](const Layout& in) {
    return in;
  });
}

/*! \brief Fix the input layout as the previous inferred (if any) and do not define output */
template<int n_in, int n_out>
inline bool ElemwiseFixedLayoutUnknownOut(const NodeAttrs& attrs,
                                          std::vector<Layout> *in_layouts,
                                          const std::vector<Layout> *last_in_layouts,
                                          std::vector<Layout> *out_layouts) {
  return ElemwiseFixedLayout<n_in, n_out>(
    attrs, in_layouts, last_in_layouts, out_layouts, [](const Layout& in) {
    return Layout::Undef();
  });
}

/*! \brief take arbitrary input layout and copy to output */
template<int n_in, int n_out>
inline bool ElemwiseArbitraryLayout(const NodeAttrs& attrs,
                                    std::vector<Layout> *in_layouts,
                                    const std::vector<Layout> *last_in_layouts,
                                    std::vector<Layout> *out_layouts) {
  const size_t in_size = (n_in == -1) ? in_layouts->size() : static_cast<size_t>(n_in);
  const size_t out_size = (n_out == -1) ? out_layouts->size() : static_cast<size_t>(n_out);

  Layout in;
  for (size_t i = 0; i < in_size; ++i) {
    if (!in.defined()) in = in_layouts->at(i);
    VERIFY_EQ(in, in_layouts->at(i))
      << "Incompatible attr in node " << attrs.name << " at " << i
      << "-th input: expected " << in
      << ", got " << in_layouts->at(i);
  }

  if (in.defined()) {
    for (size_t i = 0; i < out_size; ++i) {
      out_layouts->at(i) = in;
    }
  }

  return true;
}

/*!
 * \brief try to convert right layout to left layout if they are different.
 *        if the converting fails, it will use the last inferred layouts.
 */
inline bool ElemwiseBinaryKeepLeftLayout(const NodeAttrs& attrs,
                                         std::vector<Layout> *in_layouts,
                                         const std::vector<Layout> *last_in_layouts,
                                         std::vector<Layout> *out_layouts) {
  VERIFY_EQ(in_layouts->size(), 2U);
  VERIFY_EQ(last_in_layouts->size(), 2U);
  VERIFY_EQ(out_layouts->size(), 1U);

  const Layout& lhs_last = (*last_in_layouts)[0];
  const Layout& rhs_last = (*last_in_layouts)[1];
  VERIFY((lhs_last.defined() && rhs_last.defined()) ||
        (!lhs_last.defined() && !rhs_last.defined()));

  const Layout& lhs = (*in_layouts)[0];
  const Layout& rhs = (*in_layouts)[1];

  if (!lhs.defined() && !rhs.defined()) {
    VERIFY(!lhs_last.defined() && !rhs_last.defined())
      << "Lost input layouts in node " << attrs.name
      << ": last inferred lhs=" << lhs_last << ", rhs=" << rhs_last;
    return true;
  } else if (!lhs.defined()) {
    VERIFY(!lhs_last.defined() && !rhs_last.defined());
    in_layouts->at(0) = rhs;
    out_layouts->at(0) = rhs;
    return true;
  } else if (!rhs.defined()) {
    VERIFY(!lhs_last.defined() && !rhs_last.defined());
    in_layouts->at(1) = lhs;
    out_layouts->at(0) = lhs;
    return true;
  }

  if (lhs == rhs) {
    // for same layout, we can always do binary calculation
    // and pass the layout to next layer
    out_layouts->at(0) = lhs;
    return true;
  }

  if (rhs.convertible(lhs)) {
    in_layouts->at(1) = lhs;
    out_layouts->at(0) = lhs;
  } else {
    VERIFY(lhs_last.defined() && rhs_last.defined())
      << "Incompatible input layouts in node " << attrs.name
      << ". lhs: " << lhs << ", rhs: " << rhs;
    VERIFY(lhs_last == rhs_last);
    in_layouts->at(0) = lhs_last;
    in_layouts->at(1) = rhs_last;
    out_layouts->at(0) = lhs_last;
  }

  return true;
}

#define CVM_REGISTER_ELEMWISE_UNARY_OP(name)                       \
  CVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)        \
  .set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)           \
  .set_attr<FCorrectLayout>("FCorrectLayout",                       \
    ElemwiseArbitraryLayout<1, 1>)                                  \
  .set_attr<FInplaceOption>("FInplaceOption",                       \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "Tensor", "The input tensor.")


#define CVM_REGISTER_INIT_OP(name)                                 \
  CVM_REGISTER_OP(name)                                            \
  .set_num_inputs(0)                                                \
  .set_num_outputs(1)


#define CVM_REGISTER_INIT_LIKE_OP(name)                            \
  CVM_REGISTER_ELEMWISE_UNARY_OP(name)                             \
  .add_argument("data", "Symbol", "The input")


#define CVM_REGISTER_ELEMWISE_BINARY_OP(name)                      \
  CVM_REGISTER_OP(name)                                            \
  .set_num_inputs(2)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<FInferShape>("FInferShape", ElemwiseShape<2, 1>)        \
  .set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)           \
  .set_attr<FCorrectLayout>("FCorrectLayout",                       \
    ElemwiseBinaryKeepLeftLayout)                                   \
  .set_attr<FInplaceOption>("FInplaceOption",                       \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};     \
    })                                                              \
  .add_argument("lhs", "Tensor", "first input")                     \
  .add_argument("rhs", "Tensor", "second input")


#define CVM_REGISTER_ELEMWISE_REDUCE_OP(name)                      \
  CVM_REGISTER_OP(name)                                            \
  .set_num_inputs([](const NodeAttrs& attrs) {                      \
    return static_cast<uint32_t>(                                   \
      utils::get<ElementWiseReduceParam>(attrs.parsed).num_args);    \
    })                                                              \
  .set_attr_parser(ParamParser<ElementWiseReduceParam>)             \
  .set_attr<FGetAttrDict>("FGetAttrDict",                           \
    ParamGetAttrDict<ElementWiseReduceParam>)                       \
  .set_attr<cvm::FInferShape>("FInferShape",                       \
    ElementWiseReduceShape)                                         \
  .set_attr<FCorrectLayout>("FCorrectLayout",                       \
    ElemwiseFixedLayoutCopyToOut<-1, 1>)                             \
  .set_attr<cvm::FInferType>("FInferType", ElementWiseReduceType)  \
  .add_argument("args", "Symbol[]", "Positional input arguments")


#define CVM_REGISTER_INDICATOR_OP(name)                            \
  CVM_REGISTER_OP(name)                                            \
  .set_num_outputs(1)                                               \
  .set_attr<FInferType>(                                            \
    "FInferType", [](const NodeAttrs& attrs,                        \
                     std::vector<int>* in_attrs,                    \
                     std::vector<int>* out_attrs) {                 \
      VERIFY_EQ(out_attrs->size(), 1U);                              \
      CVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0,                 \
        static_cast<int>(kInt32));                                  \
      return true;                                                  \
  })                                                                \
  .set_attr<FCorrectLayout>("FCorrectLayout",                       \
    ElemwiseFixedLayoutUnknownOut<1, 1>)                            \


}  // namespace top
}  // namespace cvm
#endif  // CVM_TOP_ELEMWISE_OP_COMMON_H_
