/*!
 *  Copyright (c) 2017 by Contributors
 * \file op_common.h
 * \brief Common operator utilities
 */
#ifndef CVM_TOP_OP_COMMON_H_
#define CVM_TOP_OP_COMMON_H_

#include <utils/logging.h>
#include <utils/parameter.h>
#include <cvm/top/tensor.h>
#include <cvm/node.h>
#include <cvm/op_attr_types.h>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>

namespace cvm {
namespace top {

const int32_t ATTR_MIN_VALUE = 0;
const int32_t ATTR_MAX_VALUE = 4096;

/*
 * verify attribute value range
 */
template<typename T>
inline void VerifyAttrRange(
    const T& val, 
    const std::string& name, 
    const int32_t min = ATTR_MIN_VALUE, 
    const int32_t max = ATTR_MAX_VALUE) {
  VERIFY(min <= val && val <= max)
    << "attribute " << name 
    << " value: " << val
    << " out of range [" << min << ", " << max << "]";
}

/*!
 * \brief Parse keyword arguments as PType arguments and save to parsed
 * \tparam PType the parameter type.
 * \param attrs The attributes.
 */
template<typename PType>
inline void ParamParser(cvm::NodeAttrs* attrs) {
  PType param;
  try {
    param.Init(attrs->dict);
  } catch (const utils::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw utils::ParamError(os.str());
  }
  attrs->parsed = std::move(param);
}

/*!
 * \brief Parse keyword arguments as PType arguments and save to parsed
 * \tparam PType the arameter type.
 * \param attrs The attributes.
 */
template<typename PType>
inline std::unordered_map<std::string, std::string>
ParamGetAttrDict(const cvm::NodeAttrs& attrs) {
  std::unordered_map<std::string, std::string> dict = attrs.dict;
  cvm::get<PType>(attrs.parsed).UpdateDict(&dict);
  return dict;
}

/*! \brief check if shape is empty or contains unkown (0) dim. */
inline bool shape_is_none(const TShape& x) {
  return x.ndim() == 0 || x.Size() == 0;
}

/*! \brief check if type is none (-1) */
inline bool type_is_none(const int& x) {
  return x == -1;
}

/*! \brief check if shape is scalar({1}). */
inline bool shape_is_scalar(const TShape& x) {
  return x.ndim() == 1 && x.Size() == 1;
}

/*! \brief get string representation of shape */
inline std::string shape_string(const TShape& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

/*! \brief get string representation of shape */
inline std::string type_string(const int& x) {
  return std::to_string(x);
}

/*!
 * \brief Assign x to y. Checks for compatiblity when y is not empty.
 *  Allow missing dim in both x and y (as 0).
 * \param y target shape.
 * \param x source shape.
 * \return whether x and y are compatible.
 */
inline bool shape_assign(TShape *y, const TShape& x) {
  if (y->ndim() == 0) {
    *y = x;
    return true;
  } else if (y->ndim() != x.ndim()) {
    return x.ndim() == 0;
  } else {
    for (size_t i = 0; i < y->ndim(); ++i) {
      if ((*y)[i] == 0) {
        (*y)[i] = x[i];
      } else if ((*y)[i] != x[i] && x[i] != 0) {
        return false;
      }
    }
    return true;
  }
}

/*!
 * \brief Assign x to y. Checks for compatiblity when y is not -1.
 * \param y target type.
 * \param x source type.
 * \return whether x and y are compatible.
 */
inline bool type_assign(int *y, const int& x) {
  if (*y == -1) {
    *y = x;
    return true;
  } else if (*y != x && x != -1) {
    return false;
  }
  return true;
}

template<typename AttrType>
inline std::string attr_assign_error_msg(const NodeAttrs& attrs,
                                         int index, bool is_input,
                                         const AttrType& expected,
                                         const AttrType& actual,
                                         const char* attr_name) {
  static const auto& flist_inputs = Op::GetAttr<FListInputNames>("FListInputNames");
  static const auto& flist_outputs = Op::GetAttr<FListOutputNames>("FListOutputNames");
  const auto& flist = is_input ? flist_inputs : flist_outputs;
  std::string name;
  if (flist.count(attrs.op)) {
    name = flist[attrs.op](attrs)[index];
  } else {
    name = (is_input ? "data" : "output") + std::to_string(index);
  }
  std::ostringstream msg;
  msg << "Operator " << attrs.op->name << "(";
  for (const auto& kv : attrs.dict) msg << kv.first << "=" << kv.second << ", ";
  msg << "name=" << attrs.name << ") expects " << name << "\'s " << attr_name
      << " to be " << expected << ", but got " << actual << ".";
  return msg.str();
}

/*!
 * \brief macro assign shape to input if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define CVM_ASSIGN_INPUT_SHAPE(attrs, inputs, index, shape)             \
  {                                                                      \
    if (!shape_assign(&(inputs)[index], TShape(shape))) {                \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, true, shape,     \
                                          (inputs)[index], "shape");     \
    }                                                                    \
  }

/*!
 * \brief macro assign shape to out if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define CVM_ASSIGN_OUTPUT_SHAPE(attrs, outputs, index, shape)           \
  {                                                                      \
    if (!shape_assign(&(outputs)[index], TShape(shape))) {               \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, false, shape,    \
                                          (outputs)[index], "shape");    \
    }                                                                    \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define CVM_ASSIGN_INPUT_TYPE(attrs, inputs, index, type)               \
  {                                                                      \
    if (!type_assign(&(inputs)[index], type)) {                          \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, true, type,      \
                                          (inputs)[index], "type");      \
    }                                                                    \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define CVM_ASSIGN_OUTPUT_TYPE(attrs, outputs, index, type)             \
  {                                                                      \
    if (!type_assign(&(outputs)[index], type)) {                         \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, false, type,     \
                                          (outputs)[index], "type");     \
    }                                                                    \
  }

#define CVM_ASSIGN_LAYOUT(outputs, index, layout)                       \
  {                                                                      \
    if (layout.defined()) {                                              \
      (outputs)[index] = layout;                                         \
    }                                                                    \
  }

/*!
 * \brief macro assign rhs shape to lhs
 *  Use macro so we can see the error file more clearly
 * \param lhs lhs shape
 * \param rhs rhs shape
 */
#define SHAPE_ASSIGN(lhs, rhs)                                \
  if ((lhs).ndim() == 0) (lhs) = (rhs);                       \
  else                                                        \
    CHECK_EQ(lhs, rhs) << "shape inference inconsistent";     \

/*!
 * \brief macro assign rhs type to lhs
 *  Use macro so we can see the error file more clearly
 * \param lhs lhs type
 * \param rhs rhs type
 */
#define DTYPE_ASSIGN(lhs, rhs)                                \
  if ((lhs) == -1) (lhs) = (rhs);                             \
  else                                                        \
    CHECK_EQ(lhs, rhs) << "type inference inconsistent";     \

/*!
 * \brief macro check attributes precision valid
 *  Use macro so we can see the error file more clearly
 * \param attr input attributes
 * \param msg log message
 */
#define IN_PREC_CHECK(attr, op_name)                  \
  for (size_t i = 0; i < attr->size(); ++i) {         \
    VERIFY_NE(attr->at(i), -1)                        \
    << "operator " << op_name                         \
    << "'s inputs(" << i << ")"                       \
    << " has not been infered precision";             \
  }

/**
 * Calculate the sum bit of size K for Int(p) addition
 *  the reduction operator's precision is 
 *    p + ceil(log(K))
 **/
inline int GetReduceSumBit(int64_t size) {
  int prec = 0;
  while (size != 0) { prec ++; size >>= 1; }
  return prec;
}

/**
 * Calculate the precision of number
 *  number `n` of Int(p) indicates equation:
 *    abs(n) <= 2 ^ (p - 1) - 1,
 *  which is equal to transformation:
 *    log(n + 1) + 1 <= p,
 *  p is an interger, that is 
 *    p >= ceil(log(n+1)) + 1
 *  
 **/
inline int GetNumberPrecision(int64_t n) {
  int prec = 0;
  n = n + 1;
  while (n != 0) { prec++; n >>= 1; }
  return prec + 1;
}

// simply return the shape as same
inline bool SameShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  if (ishape->size() == 0 || (*ishape)[0].ndim() == 0) return false;
  for (TShape& pshape : *oshape) {
    pshape = (*ishape)[0];
  }
  for (TShape& pshape : *ishape) {
    pshape = (*ishape)[0];
  }
  return true;
}

// return shape from node attrs
template<typename PType>
inline bool ZeroShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  const TShape& ts = utils::get<PType>(attrs.parsed).shape;
  if (ts.ndim() != 0) {
    SHAPE_ASSIGN(oshape->at(0), ts);
    return true;
  } else {
    return false;
  }
}

// do not infer layout
inline bool ZeroLayout(const NodeAttrs& attrs,
                       std::vector<Layout> *in_layouts,
                       const std::vector<Layout> *last_in_layouts,
                       std::vector<Layout> *out_layouts) {
  return true;
}

// simply assign output shape or type from input
template<typename AttrType, int in_index, int out_index>
inline bool AssignOutputAttr(const NodeAttrs& attrs,
                              std::vector<AttrType> *in_attrs,
                              std::vector<AttrType> *out_attrs) {
  CHECK_LT(in_index, in_attrs->size());
  CHECK_LT(out_index, out_attrs->size());
  const TShape &dshape = in_attrs->at(in_index);
  CVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, out_index, dshape);
  return true;
}

// return type from node attrs
template<typename PType>
inline bool ZeroType(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int dtype = utils::get<PType>(attrs.parsed).dtype;
  DTYPE_ASSIGN(oattr->at(0), dtype);
  return true;
}

// Make zero grad node
inline std::vector<NodeEntry> MakeZeroGradNodes(
  const NodePtr& n,
  const std::vector<NodeEntry>& ograds) {
  std::vector<NodeEntry> ret;
  for (uint32_t i = 0; i < n->num_inputs(); ++i) {
    std::ostringstream os;
    ret.push_back(MakeNode("zeros_like", n->attrs.name + "_zero_grad",
                           {n->inputs[i]}));
  }
  return ret;
}

// Helper to make gradient node
inline std::vector<NodeEntry> MakeGradNode(
  const char* op_name,
  const NodePtr& n,
  std::vector<NodeEntry> inputs,
  std::unordered_map<std::string, std::string> attr = {{}}) {
  NodePtr p = Node::Create();
  p->attrs.op = cvm::Op::Get(op_name);
  p->attrs.name = n->attrs.name + "_grad";
  p->inputs = std::move(inputs);
  p->attrs.dict = std::move(attr);
  if (p->attrs.op->attr_parser) {
    p->attrs.op->attr_parser(&p->attrs);
  }
  std::vector<NodeEntry> ret;
  for (uint32_t i = 0; i < p->num_outputs(); ++i) {
    ret.emplace_back(NodeEntry{p, i, 0});
  }
  return ret;
}


}  // namespace top
}  // namespace cvm

#endif  // CVM_TOP_OP_COMMON_H_
