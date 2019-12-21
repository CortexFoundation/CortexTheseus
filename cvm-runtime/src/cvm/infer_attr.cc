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
  for (auto shape : shapes) ret.emplace_back(shape);
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
      VERIFY_NE(precision[entry_id(nid, 0)], -1)
        << "variable node " << inode.name()
        << "'s precision has not been set";
    } else {
      const uint32_t num_inputs = inode.param.num_inputs;
      const uint32_t num_outputs = inode.param.num_outputs;
      // Forward operator inference.
      shapes.resize(num_inputs+num_outputs, TShape());
      iprec.resize(num_inputs, -1);
      oprec.resize(num_outputs, -1);
      for (uint32_t i = 0; i < num_inputs; ++i) {
        const auto& eid = entry_id(inode.inputs[i]);
        iprec[i] = precision[eid];
        shapes[i] = rshape[eid];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        const auto& eid = entry_id(nid, i);
        oprec[i] = precision[eid];
        shapes[num_inputs+i] = rshape[eid];
      }
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
        VERIFY((0 < oprec[i]) && (oprec[i] <= 32))
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
  int64_t MAX_BASE_OPS = (int64_t)1 << 30;
  int64_t MAX_OPS = ((int64_t)1 << 40);
  int64_t MAX_MEMORY = ((int64_t)1 << 40);
  for (uint32_t nid = 0; nid < idx.size(); ++nid) {
    auto inode = idx[nid];
    if (inode.is_variable()) {
      int64_t mem_size = rshape[entry_id(nid, 0)].Size();
      mem_cost += mem_size * 5;
    } else {
      uint32_t out_eid = entry_id(nid, 0);
      int64_t base_ops = 1;
      auto& op_name = inode.op()->name;
      if (op_name == "dense") {
        auto& param = cvm::get<cvm::top::DenseParam>(inode.attrs.parsed);
        auto weight_shp = rshape[entry_id(inode.inputs[param.kWeight])];
        base_ops = static_cast<int64_t>(weight_shp[1]) * 3; // MAX (1<<24) * 3 < 1G
        if (param.use_bias) base_ops += 1;
      } else if (op_name == "non_max_suppression") {
        auto shape = rshape[entry_id(inode.inputs[0])];
        base_ops = static_cast<int64_t>(shape[0]) * 20;
      } else if (op_name == "conv2d") {
        auto& param = cvm::get<cvm::top::Conv2DParam>(inode.attrs.parsed);
        auto weight_shp = rshape[entry_id(inode.inputs[param.kWeight])];
        base_ops = weight_shp.Size() / weight_shp[0] * 3;
        if (param.use_bias) base_ops += 1;
      } else if (op_name == "max_pool2d") {
        auto& param = cvm::get<cvm::top::MaxPool2DParam>(inode.attrs.parsed);
        base_ops = param.pool_size.Size(); // MAX pool_size < input shape < 1G
      } else if (op_name == "sum") {
        auto input_shp = rshape[entry_id(inode.inputs[0])];
        base_ops = input_shp.Size() / rshape[out_eid].Size(); // MAX < 1G
      } else if (op_name == "get_valid_count") {
        // operator output is `valid_count` and `output array`,
        // so use the index 1 as the main output entry id.
        out_eid = entry_id(nid, 1);
      }
      VERIFY_LE(base_ops, MAX_BASE_OPS)
        << "single ops foreach output should not greater than 1G"
        << ", but " << base_ops;

      int64_t osize = rshape[out_eid].Size(); // output size < 1G
      base_ops *= osize; // MAX 1G * 1G < (1 << 60);
      ops += base_ops; // MAX (1 << 40) + (1 << 60) < int64_t
      VERIFY_LE(ops, MAX_OPS) << "graph ops exceed MAX_OPS " << MAX_OPS;

      // Calculate internal symbol's memory cost with output shape,
      // which multiply scale 5 by default.
      int64_t mem_size = 0; // MAX (1 << 40) * num_outputs < (1 << 41)
      for (uint32_t i = 0; i < inode.param.num_outputs; ++i) {
        mem_size += rshape[entry_id(nid, i)].Size();
      }
      mem_cost += mem_size * 5; // MAX (1 << 40) + (1 << 44) < int64_t
    }
    VERIFY_LE(mem_cost, MAX_MEMORY)
      << "graph memory cost exceed MAX_MEMORY " << MAX_MEMORY;
  }
  int64_t ret = mem_cost + ops;
  std::cout << "GetOps: memory cost=" << int(mem_cost / 1000000)
    << "M percentage=" << 1.f * mem_cost / (ret + 1e-5)
    << " ops=" << int(ops / 1000000)
    << "M percentage=" << 1.f * ops / (ret + 1e-5) << std::endl;
  return mem_cost + ops; // (1 << 40) + (1 << 32) < int64_t
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
  // InferType use all "int32", to simplify verify.
  // More type check use InferPrecision.
  auto &idx = nodes_;
  std::vector<int> rtype(this->num_node_entries_, 4);
  // std::vector<std::string> &dltype = attrs_.dltype;
  // for (unsigned int i = 0; i < dltype.size(); ++i) {
  //   VERIFY_EQ(dltype[i], "int32")
  //     << "type " << dltype[i] << " are not supported.";
  //   rtype.push_back(4);
  // }

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
