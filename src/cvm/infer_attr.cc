/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_attr.cc
 * \brief Inference the attrs given existin information.
 */
#include "graph_runtime.h"
#include <cvm/op.h>
#include <cvm/op_attr_types.h>
#include "top/elemwise_op_common.h"
#include <cvm/graph_attr_types.h>

#include <iostream>

using cvm::Op;
using cvm::TShape;

namespace cvm {
namespace runtime {

bool CvmRuntime::CheckAttr() {
  SetupShape();
  SetupType();
  SetupPrecision();
  return true;
}

std::vector<TShape> GetTShapeArray(const std::vector<std::vector<int64_t> > &shapes) {
  std::vector<TShape> ret;
  for (auto shape : shapes) {
    if (shape.size() == 1) {
      ret.push_back(TShape{shape[0]});
    } else if (shape.size() == 2) {
      ret.push_back(TShape{shape[0], shape[1]});
    } else if (shape.size() == 3) {
      ret.push_back(TShape{shape[0], shape[1], shape[2]});
    } else if (shape.size() == 4) {
      ret.push_back(TShape{shape[0], shape[1], shape[2], shape[3]});
    } else {
      ret.push_back(TShape());
    }
  }
  return ret;
}

void CvmRuntime::SetupPrecision() {
  std::vector<Node> &idx = nodes_;
  std::vector<int> &precision = attrs_.precision;
  const auto rshape = GetTShapeArray(attrs_.shape);
  precision.resize(nodes_.size(), -1);
  // Temp space for shape inference.
  std::vector<int> iprec, oprec;
  std::vector<TShape> shapes;
  static auto& finfer_prec =
      Op::GetAttr<cvm::FInferPrecision>("FInferPrecision");
 
  // inference step function for nid
  auto infer_prec = [&](uint32_t nid) {
    const auto& inode = idx[nid];
    if (inode.op_type == "null") {
      if (precision[nid] == -1) {
         precision[nid] = 8;
      }
      // Variable node. No operator. Only one output entry.
    } else {
      const uint32_t num_inputs = inode.param.num_inputs;
      const uint32_t num_outputs = inode.param.num_outputs;
      // Forward operator inference.
      iprec.resize(num_inputs, -1);
      shapes.resize(num_inputs, TShape());
      for (uint32_t i = 0; i < iprec.size(); ++i) {
        iprec[i] = precision[entry_id(inode.inputs[i])];
        shapes[i] = rshape[entry_id(inode.inputs[i])];
      }
      VERIFY_GE(num_outputs, 1) << "an operator has at least 1 outputs";
      oprec.resize(num_outputs, -1);
      auto finfer = finfer_prec.get(inode.attrs.op, nullptr);
      // Call inference function of the operator.
      if (finfer == nullptr) {
        std::cout << "no infer precision method " << inode.attrs.op->name << std::endl;
        finfer = cvm::top::ElemwiseSamePrecision;  
      }
      if (!finfer(inode.attrs, &shapes, &iprec, &oprec)) {
        throw utils::Error(std::string("error with ") + inode.attrs.op->name);
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
          VERIFY(precision[entry_id(inode.inputs[i])] <= iprec[i] && iprec[i] <= 32)
             << "Check precision failed, "
             << "expected to be at most " << iprec[i]
             << " but " << precision[entry_id(inode.inputs[i])];
          precision[entry_id(inode.inputs[i])] = iprec[i];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        precision[entry_id(nid, i)] = oprec[i];
      }
    }
  };

  for (uint32_t nid = 0; nid < idx.size(); ++nid) {
    infer_prec(nid);
  }
}

int64_t CvmRuntime::GetOps() {
  auto &idx = nodes_;
  const auto rshape = GetTShapeArray(attrs_.shape);
  // inference step function for nid
  int64_t ret = 0;
  std::vector<std::string> ops;
  std::unordered_map<std::string, int64_t> opcount;
  for (uint32_t nid = 0; nid < idx.size(); ++nid) {
    auto inode = idx[nid];
    if (inode.op_type == "null") {
      int64_t osize = rshape[nid].Size();
      osize /= rshape[nid][0];
      ret += osize;
    } else {
      auto op = idx[nid].attrs.op->name;
      if (opcount.find(op) == opcount.end()) {
        opcount[op] = 0;
        ops.push_back(op);
      }

      int64_t t = 0;
      if (op == "dense") {
        auto shape1 = rshape[inode.inputs[0].node_id];
        auto shape2 = rshape[inode.inputs[1].node_id];
        t = static_cast<int64_t>(shape2[1]) * 2;
        auto& param = cvm::get<cvm::top::DenseParam>(inode.attrs.parsed);
        if (param.use_bias) {
          t += 1;
        }
      } else if (op == "conv2d") {
        auto shape1 = rshape[inode.inputs[0].node_id];
        auto shape2 = rshape[inode.inputs[1].node_id];
        t = (static_cast<int64_t>(shape2[1]) * shape2[2] * shape2[3] * 2);
        auto& param = cvm::get<cvm::top::Conv2DParam>(inode.attrs.parsed);
        if (param.use_bias) {
          t += 1;
        }
     } else if (op == "max_pool2d") {
        t = rshape[nid].Size();
        t /= rshape[nid][0];
        auto& param = cvm::get<cvm::top::MaxPool2DParam>(inode.attrs.parsed);
        t = param.pool_size.Size();
      } else {
        t = 1;
      }
      int64_t osize = rshape[nid].Size();
      osize /= rshape[nid][0];
      t *= osize;
/*
      std::cout << op << "    ";
      for (int i = op.length(); i < 20; ++i) std::cout << ' ';
      for (auto n : inode.inputs) {
        std::cout << rshape[n.node_id] << "    ";
      }
      std::cout << rshape[nid] << ' ' << t << std::endl;
*/
      ret += t;
      opcount[op] += t;
    }
  }
  return ret;
}

void CvmRuntime::SetupShape() {
  auto &idx = nodes_;
  const auto rshape = GetTShapeArray(attrs_.shape);
  static auto& finfer_shape =
      Op::GetAttr<cvm::FInferNodeEntryAttr<TShape> >("FInferShape");
  // reshape shape vector
  // Temp space for shape inference.
  std::vector<TShape> ishape, oshape;

  // inference step function for nid
  auto infer_shape = [&](uint32_t nid) {
    const auto& inode = idx[nid];
    if (inode.op_type == "null") {
      // Variable node. No operator. Only one output entry.
      VERIFY(rshape[nid].ndim()) << "Invalid variable shape";
    } else {
      const uint32_t num_inputs = inode.param.num_inputs;
      const uint32_t num_outputs = inode.param.num_outputs;
      // Forward operator inference.
      ishape.resize(num_inputs, TShape());
      for (uint32_t i = 0; i < ishape.size(); ++i) {
        ishape[i] = rshape[entry_id(inode.inputs[i])];
      }
      oshape.resize(num_outputs, TShape());
      for (uint32_t i = 0; i < oshape.size(); ++i) {
        oshape[i] = TShape();
      }
      // which raise an error if the op has not been registered.
      auto finfer = finfer_shape.get(inode.attrs.op, nullptr);
      if (finfer != nullptr) {
        // Call inference function of the operator.
        try {
          finfer(inode.attrs, &ishape, &oshape);
        } catch (const std::exception& e) {
          throw utils::Error(e.what() + std::string(" with ") + inode.attrs.op->name);
        }
      } else {
        throw utils::Error(std::string("check shape method is undefined with") + inode.attrs.op->name);
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        VERIFY_EQ(ishape[i], rshape[entry_id(inode.inputs[i])])
          << "Check input shape failed, "
          << "expected to be " << ishape[i]
          << " but " << rshape[entry_id(inode.inputs[i])];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        VERIFY_EQ(oshape[i], rshape[entry_id(nid, i)])
          << "Check output shape failed, "
          << "expected to be " << oshape[i]
          << " but " << rshape[entry_id(nid, i)];
      }
    }
  };

  for (uint32_t nid = 0; nid < idx.size(); ++nid) {
    infer_shape(nid);
  }
}

// inference fucntion for same type
inline bool SameType(const cvm::NodeAttrs attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int def_v = -1;
  for (int v : *oattr) {
    if (v != -1) {
      def_v = v; break;
    }
  }
  if (def_v == -1) {
    for (int v : *iattr) {
      if (v != -1) {
        def_v = v; break;
      }
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    v = def_v;
  }
  return true;
}

void CvmRuntime::SetupType() {
  auto &idx = nodes_;
  std::vector<int> rtype;
  rtype.resize(nodes_.size(), 4);
  static auto& finfer_type =
      Op::GetAttr<cvm::FInferNodeEntryAttr<int> >("FInferType");
  // reshape shape vector

  // Temp space for shape inference.
  std::vector<int> itype, otype;

  // inference step function for nid
  auto infer_type = [&](uint32_t nid) {
    const auto& inode = idx[nid];
    if (inode.op_type == "null") {
      // Variable node. No operator. Only one output entry.
    } else {
      const uint32_t num_inputs = inode.param.num_inputs;
      const uint32_t num_outputs = inode.param.num_outputs;
      // Forward operator inference.
      itype.resize(num_inputs, -1);
      for (uint32_t i = 0; i < num_inputs; ++i) {
        itype[i] = rtype[entry_id(inode.inputs[i])];
      }
      otype.resize(num_outputs, -1);
      for (uint32_t i = 0; i < num_outputs; ++i) {
        otype[i] = -1;
      }
      // which raise an error if the op has bit been registered.
      auto finfer = finfer_type.get(inode.attrs.op, SameType);
      if (finfer != nullptr) {
        try {
          cvm::NodeAttrs attrs;
          finfer(inode.attrs, &itype, &otype);
        } catch (const std::exception& e) {
          throw utils::Error(e.what() + std::string(" with ") + inode.attrs.op->name);
        }
      } else {
        throw utils::Error(std::string("check type method is undefined with") + inode.attrs.op->name);
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        VERIFY_EQ(itype[i], rtype[entry_id(inode.inputs[i])])
          << "Check type failed, "
          << "expected to be " << itype[i]
          << " but " << rtype[entry_id(inode.inputs[i])];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        VERIFY_EQ(otype[i], rtype[entry_id(nid, i)])
          << "Check type failed, "
          << "expected to be " << otype[i]
          << " but " << rtype[entry_id(nid, i)];
      }
    }
  };

  for (uint32_t nid = 0; nid < idx.size(); ++nid) {
    infer_type(nid);
  }
}

}
}
