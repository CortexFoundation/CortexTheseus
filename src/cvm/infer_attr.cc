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
    VERIFY_LE(shape.size(), 6) << "shape size should not larger than 6";
    for (size_t i = 0; i < shape.size(); ++i) {
      VERIFY_LE(shape[i], 0x7fffffff)
        << "tensor size should not larger than the range of int32";
    }
    ret.emplace_back(shape);
  }
  return ret;
}

void CvmRuntime::SetupPrecision() {
  std::vector<Node> &idx = nodes_;
  std::vector<int> &precision = attrs_.precision;
  const auto rshape = GetTShapeArray(attrs_.shape);
  std::vector<int> iprec, oprec;
  std::vector<TShape> shapes;
  static auto& finfer_prec =
      Op::GetAttr<cvm::FInferPrecision>("FInferPrecision");

  // inference step function for nid
  auto infer_prec = [&](uint32_t nid) {
    const auto& inode = idx[nid];
    if (inode.op_type == "null") {
      // Variable node. No operator. Only one output entry.
      const auto& eid = entry_id(nid, 0);
      VERIFY_NE(precision[eid], -1)
        << "variable node " << inode.name()
        << "'s precision has not been set";
      VERIFY_LE(precision[eid], 32)
        << "variable node " << inode.name()
        << "'s precision out of INT32";
    } else {
      const uint32_t num_inputs = inode.param.num_inputs;
      const uint32_t num_outputs = inode.param.num_outputs;
      // Forward operator inference.
      iprec.resize(num_inputs, -1);
      shapes.resize(num_inputs, TShape());
      for (uint32_t i = 0; i < iprec.size(); ++i) {
        const auto& eid = entry_id(inode.inputs[i]);
        iprec[i] = precision[eid];
        shapes[i] = rshape[eid];
      }
      VERIFY_GE(num_outputs, 1) << "an operator has at least 1 outputs";
      oprec.resize(num_outputs, -1);
      auto finfer = finfer_prec.get(inode.attrs.op, nullptr);
      // Call inference function of the operator.
      if (finfer == nullptr) {
        LOG(FATAL) << "operator " << inode.op()->name
          << " has not registered FInferPrecision";
      }
      if (!finfer(inode.attrs, &shapes, &iprec, &oprec)) {
        LOG(FATAL) << "operator " << inode.op()->name
          << " name=" << inode.name()
          << ": infer precision failed";
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_outputs; ++i) {
        precision[entry_id(nid, i)] = oprec[i];
        VERIFY_LE(oprec[i], 32)
            << " nid = " << nid << "i = " << i
            << " precison = " << oprec[i]
            << " name= " << inode.name()
            << " inode.attrs = " << attrs_.op_attrs[nid];
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
  int64_t ops = 0, mem_cost = 0;
  for (uint32_t nid = 0; nid < idx.size(); ++nid) {
    auto inode = idx[nid];
    if (inode.op_type == "null") {
      int64_t mem_size = rshape[entry_id(nid, 0)].Size();
      mem_cost += mem_size * 5;
    } else {
      uint32_t out_eid = entry_id(nid, 0);
      int64_t t = 1;
      int len = 0;
      auto op = idx[nid].attrs.op->name;
      if (op == "dense") {
        auto shape1 = rshape[entry_id(inode.inputs[1])];
        VERIFY_GE(shape1.ndim(), 2);
        t = static_cast<int64_t>(shape1[1]) * 3;
        auto& param = cvm::get<cvm::top::DenseParam>(inode.attrs.parsed);
        if (param.use_bias) {
          t += 1;
        }
        len += 32 - __builtin_clz(unsigned(t));
      } else if (op == "non_max_suppression") {
        auto shape1 = rshape[entry_id(inode.inputs[0])];
        VERIFY_GE(shape1.ndim(), 1);
        t = static_cast<int64_t>(shape1[0]) * 20;
        len += 32 - __builtin_clz(unsigned(t));
      } else if (op == "conv2d") {
        auto shape1 = rshape[entry_id(inode.inputs[0])];
        auto shape2 = rshape[entry_id(inode.inputs[1])];
        VERIFY_GE(shape1.ndim(), 4);
        VERIFY_GE(shape2.ndim(), 4);
        t = (static_cast<int64_t>(shape2[1]) * shape2[2] * shape2[3] * 3);
        len += 96 - __builtin_clz((unsigned)shape2[1]) - __builtin_clz((unsigned)shape2[2])
                  - __builtin_clz((unsigned)shape2[3] * 3);
        auto& param = cvm::get<cvm::top::Conv2DParam>(inode.attrs.parsed);
        if (param.use_bias) {
          t += 1;
        }
      } else if (op == "max_pool2d") {
        auto& param = cvm::get<cvm::top::MaxPool2DParam>(inode.attrs.parsed);
        t = param.pool_size.Size();
        len += 32 - __builtin_clz((unsigned)t);
      } else if (op == "sum") {
        auto shape1 = rshape[entry_id(inode.inputs[0])];
        VERIFY(rshape[out_eid].Size() != 0);
        int64_t d = shape1.Size() / rshape[out_eid].Size();
        t = static_cast<int>(d);
        len += 32 - __builtin_clz((unsigned)t);
      } else if (op == "get_valid_count") {
        // operator output is `valid_count` and `output array`,
        // so use the index 1 as the main output entry id.
        out_eid = entry_id(nid, 1);
      } else {
        t = 1;
      }

      VERIFY_GE(rshape[out_eid].ndim(), 1);
      VERIFY(rshape[out_eid][0] != 0);
      int64_t osize = rshape[out_eid].Size();
      len += 32 - __builtin_clz((unsigned)osize);
      t *= osize;
      if (len > 40 || t > (1ll << 38)) {
        return -1;
      }
      ops += t;

      // Calculate internal symbol's memory cost with output shape,
      // which multiply scale 5 by default.
      int64_t mem_size = 0;
      for (uint32_t i = 0; i < inode.param.num_outputs; ++i) {
        mem_size += rshape[entry_id(nid, i)].Size();
      }
      mem_cost += mem_size * 5;
    }
  }
  int64_t ret = mem_cost + ops;
  std::cout << "GetOps: memory cost=" << int(mem_cost / 1000000)
    << "M percentage=" << 1.f * mem_cost / (ret + 1e-5)
    << " ops=" << int(ops / 1000000)
    << "M percentage=" << 1.f * ops / (ret + 1e-5) << std::endl;
  return mem_cost + ops;
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
    if (inode.is_variable()) {
      // Variable node. No operator. Only one output entry.
      VERIFY(rshape[entry_id(nid, 0)].ndim()) << "Invalid variable shape";
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
      auto finfer = finfer_shape.get(inode.op(), nullptr);
      if (finfer == nullptr) {
        LOG(FATAL) << "operator " << inode.op()->name
          << " has not registered FInferShape";
      }

      if (!finfer(inode.attrs, &ishape, &oshape)) {
        LOG(FATAL) << "operator " << inode.op()->name
          << " name=" << inode.name() << ": infer shape failed";
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
          << " but " << rshape[entry_id(nid, i)] << inode.attrs.op->name;
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
  std::vector<std::string> &dltype = attrs_.dltype;
  for (unsigned int i = 0; i < dltype.size(); ++i) {
    VERIFY_EQ(dltype[i], "int32")
      << "type " << dltype[i] << " are not supported.";
    rtype.push_back(4);
  }

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

      auto finfer = finfer_type.get(inode.attrs.op, SameType);
      if (finfer == nullptr) {
        LOG(FATAL) << "operator " << inode.op()->name
          << " has not registered FInferType";
      }
      if (!finfer(inode.attrs, &itype, &otype)) {
        LOG(FATAL) << "operator " << inode.op()->name
          << " name=" << inode.name() << ": infer type failed";
      }

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
