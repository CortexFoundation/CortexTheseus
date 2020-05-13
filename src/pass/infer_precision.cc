/*!
 *  Copyright (c) 2019 by Contributors
 * \file infer_precision.cc
 * \brief Inference the precisions given existing information.
 */
#include <cvm/pass.h>
#include <cvm/op_attr_types.h>
#include <cvm/graph_attr_types.h>
#include <unordered_map>
#include <iostream>

namespace cvm {
namespace pass {
namespace {

    /*
inline bool InferPrecisionDefault(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
    std::cout << attrs.op->name << std::endl;
    return true;
}
*/
/*
typedef bool InferFunc(std::vector<int>*, std::vector<int>*);

std::unordered_map<std::string,InferFunc> InferFuncMap({
        {
            "clip",
            [bool](std::vector<int>* iattr, std::vector<int>* oattr){
                return true;
            }
        },
        });
*/
inline bool InferPrecisionForward(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
    // std::cout << attrs.op->name << ' ' << iattr->size() << ' ' << oattr->size() << std::endl;

    return true;
}

inline bool InferPrecisionBackward(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
    std::cout << attrs.op->name << std::endl;
    return true;
}

inline bool is_none(const int t){ return t == -1; }

Graph InferPrecision(Graph &&ret) {
  using AttrVector = std::vector<int>;
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_prec =
      Op::GetAttr<cvm::FInferPrecision>("FInferPrecision");

  AttrVector precision;
  std::vector<TShape> rshape;
  rshape.resize(idx.num_node_entries(), TShape());
  precision.resize(idx.num_node_entries(), -1);

  const char* input_name = "precision_inputs";
  if (ret.attrs.count(input_name) != 0 && ret.attrs.count("shape_inputs") != 0) {
    const AttrVector& prec_args = ret.GetAttr<AttrVector>(input_name);
    for (size_t i = 0; i < prec_args.size(); ++i) {
      precision[idx.entry_id(idx.input_nodes()[i], 0)] = prec_args[i];
    }
    const std::vector<TShape>& shape_args = ret.GetAttr<std::vector<TShape>>("shape_inputs");
    for (size_t i = 0; i < shape_args.size(); ++i) {
      rshape[idx.entry_id(idx.input_nodes()[i], 0)] = shape_args[i];
    }
  } else {
    precision.resize(idx.num_node_entries(), -1);
    rshape.resize(idx.num_node_entries(), TShape());
  }

  // get the shape hints
//  std::string shape_hints_key = std::string("precision") + "_hints";
//  if (ret.attrs.count(shape_hints_key)) {
//    NodeEntryMap<int> shape_hints =
//      ret.GetAttr<NodeEntryMap<int>>(shape_hints_key);
//    for (const auto& kv : shape_hints) {
//      NodeEntry e = kv.first;
//      if (idx.exist(e.node.get())) {
//        precision[idx.entry_id(kv.first)] = kv.second;
//      }
//    }
//  }

  
  std::string shape_attr_key;
  if (ret.attrs.count("precision_attr_key") != 0) {
    shape_attr_key = ret.GetAttr<std::string>("precision_attr_key");
    // erase the provided arguments
    ret.attrs.erase("precision_attr_key");
  } else {
    shape_attr_key = "precision";
  }

  // Temp space for shape inference.
  std::vector<int> iprec, oprec;
  std::vector<TShape> shapes;

  // inference step function for nid
  auto infer_step = [&](uint32_t nid, bool last_iter) {
    const auto& inode = idx[nid];
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    if (inode.source->is_variable()) {
      // Variable node. No operator. Only one output entry.
      CHECK(inode.source->op() == nullptr);
      CHECK_EQ(num_outputs, 1U);
      const uint32_t out_ent_id = idx.entry_id(nid, 0);
//      std::cout << "Variable at " << nid << " " <<  inode.source->attrs.name << std::endl;
      if (shape_attr_key.length() != 0 && is_none(precision[out_ent_id])) {
        auto it = inode.source->attrs.dict.find(shape_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          std::istringstream is(it->second);
          CHECK(is >> rshape[out_ent_id]) << "Invalid attribute";
        }
      }
    } else {
      bool forward_known = true;
      // Forward operator inference.
      shapes.resize(num_inputs + num_outputs, TShape());
      iprec.resize(num_inputs, -1);
      for (uint32_t i = 0; i < num_inputs; ++i) {
        const auto& eid = idx.entry_id(inode.inputs[i]);
        iprec[i] = precision[eid];
        shapes[i]= rshape[eid];
        if (is_none(iprec[i])) forward_known = false;
      }
      oprec.resize(num_outputs, -1);
      for (uint32_t i = 0; i < num_outputs; ++i) {
        const auto& eid = idx.entry_id(nid, i);
        oprec[i] = precision[eid];
        shapes[num_inputs + i] = rshape[eid];
        if (is_none(oprec[i])) forward_known = false;
      }
      auto finfer = finfer_prec.get(inode.source->op(), nullptr);
      if (!forward_known) {
        if (finfer != nullptr) {
          // Call inference function of the operator.
          try {
            forward_known = finfer(inode.source->attrs, &shapes, &iprec, &oprec);
          } catch (const std::exception& e) {
            throw utils::Error("Error in operator " + inode.source->attrs.name + ": " + e.what());
          }
        } else {
          CHECK(!last_iter)
              << "Attribute " << "FInferPrecision"
              << " is not registered by op " << inode.source->op()->name
              << " we are not able to complete the inference because of this";
        }
      }
      // Save to the result map.
      //for (uint32_t i = 0; i < num_inputs; ++i) {
      //  rshape[idx.entry_id(inode.inputs[i])] = ishape[i];
      //}
      for (uint32_t i = 0; i < num_outputs; ++i) {
       // rshape[idx.entry_id(nid, i)] = oshape[i];
        precision[idx.entry_id(nid, i)] = oprec[i];
      }
    }
  };

  size_t last_num_unknown;
  size_t num_unknown = rshape.size();
  int i = 0;
  do {
    if (i % 2 == 0) {
      for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
        infer_step(nid, false);
      }
    } else {
      // backward inference
      for (uint32_t i = idx.num_nodes(); i != 0; --i) {
        infer_step(i - 1, false);
      }
    }
    last_num_unknown = num_unknown;
    num_unknown = 0;
    for (size_t j = 0; j < idx.num_node_entries(); ++j) {
      if (is_none(precision[j])) {
        ++num_unknown;
      }
    }
    ++i;
  } while (num_unknown > 0 && last_num_unknown > num_unknown);
  // set the precisions
  ret.attrs["precision"] = std::make_shared<any>(std::move(oprec));
  // number of nodes who knows the precision.
  //ret.attrs["precision_num_unknown_nodes"] = std::make_shared<any>(num_unknown);
  return std::move(ret);
}

CVM_REGISTER_PASS(InferPrecision)
.describe("Infer the precesion of each node entries.")
.set_body([](Graph ret) {
    return InferPrecision(std::move(ret));
  })
.set_change_graph(false)
.provide_graph_attr("precision");

CVMUTIL_JSON_ENABLE_ANY(size_t, size_t);

}  // namespace
}  // namespace pass
}  // namespace cvm
