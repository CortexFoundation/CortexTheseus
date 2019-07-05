/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 */
#include "graph_runtime.h"

#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/serializer.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <utility>

#define CALL_BEGIN() *rv = 0;                        \
                      try {
#define CALL_END() } \
                catch (const std::runtime_error &e) { *rv = -2; }  \
                catch (const std::logic_error &e)   { *rv = -1; }  \
              return; \


namespace cvm {
namespace runtime {

/*!
 * \brief Run all the operations one by one.
 */
void CvmRuntime::Run() {
  // setup the array and requirements.
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) op_execs_[i]();
  }
}

/*!
 * \brief Initialize the graph executor with graph and context.
 * \param graph_json The execution graph.
 * \param module The module containing the compiled functions for the host
 * processor.
 * \param ctxs The context of the host and devices where graph nodes will be
 * executed on.
 */
void CvmRuntime::SetGraph(const std::string& graph_json) {
  graph_json_ = graph_json;
}

void CvmRuntime::SetContext(const std::vector<CVMContext>& ctxs) {
  ctxs_ = ctxs;
}

void CvmRuntime::Init() {
  std::istringstream is(graph_json_);
  utils::JSONReader reader(&is);
  this->Load(&reader);
  this->PrepareGraphWithVersion();
  this->CheckAttr();
}

void CvmRuntime::PrepareGraphWithVersion() {
  // Check input node names unique
  std::unordered_set<std::string> name_set;
  for (auto nid : input_nodes_) {
    auto name = nodes_[nid].name();
    auto ret = name_set.emplace(name);
    if (!ret.second) {
      LOG(FATAL) << "node name " << name << " duplicated in graph";
    }
  }

  // CVM executor load operators
  VERIFY_EQ(nodes_.size(), attrs_.op_attrs.size())
    << "graph attribute op_attrs size: " << attrs_.op_attrs.size()
    << ", Expected " << nodes_.size();
  for (size_t i = 0; i < nodes_.size(); ++i) {
    nodes_[i].LoadOpAndAttrs(attrs_.op_attrs[i]);
  }

  num_node_entries_ = 0;
  std::vector<uint32_t> node_row_ptr;
  std::vector<uint32_t> input_nodes;
  for (uint32_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].is_variable()) input_nodes.push_back(i);
    node_row_ptr.push_back(this->num_node_entries_);
    num_node_entries_ += nodes_[i].num_outputs();
  }
  node_row_ptr.push_back(num_node_entries_);

  VERIFY_EQ(num_node_entries_, attrs_.storage_id.size())
    << "graph attribute storage_id size: " << attrs_.storage_id.size()
    << ", Expected " << num_node_entries_;
  for (auto sid : attrs_.storage_id) {
    VERIFY_GE(sid, 0) << "storage id should not less than 0, but " << sid;
  }

  // Verify shape
  VERIFY_EQ(num_node_entries_, attrs_.shape.size())
    << "graph attribute shape size: " << attrs_.shape.size()
    << ", Expected " << num_node_entries_;
  for (auto shape: attrs_.shape) {
    uint64_t sx = 1;
    VERIFY((0 < shape.size()) && (shape.size() <= 6))
      << "shape ndim should between (0, 6], but " << shape.size();
    for (auto x : shape) {
      sx *= x;
      VERIFY((0 < x) && (x <= (1<<24)))
        << "single dimension should between (0, " << (1<<24)
        << "], but " << x;
      VERIFY_LE(sx, (1<<30))
        << "shape size shoule not greater than" << (1<<30)
        << ", but " << sx;
    }
  }

  // Verify precision
  VERIFY_EQ(num_node_entries_, attrs_.precision.size())
    << "graph attribute precision size: " << attrs_.precision.size()
    << ", Expected " << num_node_entries_;
  for (auto prec : attrs_.precision) {
    VERIFY((prec == -1) || (0 < prec && prec <= 32))
      << "precision should be -1 or between (0, 32], but " << prec;
  }

  // Verify device_index if set in graph
  if (!attrs_.device_index.empty()) {
    VERIFY_EQ(num_node_entries_, attrs_.device_index.size())
      << "graph attribute device_index size: " << attrs_.device_index.size()
      << ", Expected " << num_node_entries_;
  }

  // Verify node_row_ptr and input_nodes with cvm version
  std::string& version = this->version_;
  if (version == "cvm_1.0.0") {
    VERIFY_EQ(num_node_entries_, attrs_.dltype.size())
      << "graph attribute dltype size: " << attrs_.dltype.size()
      << ", Expected " << num_node_entries_;
    for (auto dtype : attrs_.dltype) {
      VERIFY_EQ(dtype, "int32") << "type " << dtype << " are not supported";
    }

    VERIFY_EQ(node_row_ptr_.size(), node_row_ptr.size())
      << "node_row_ptr's size: " << (node_row_ptr_.size())
      << ", Expected " << node_row_ptr.size();
    for (size_t i = 0; i < node_row_ptr.size(); ++i) {
      VERIFY_EQ(node_row_ptr[i], node_row_ptr_[i])
        << "node_row_ptr is invalid at index " << i
        << " with value " << node_row_ptr_[i] << ", Expected "
        << node_row_ptr[i];
    }
    
    VERIFY_EQ(input_nodes_.size(), input_nodes.size())
      << "arg_nodes' size: " << input_nodes_.size()
      << ", Expected " << input_nodes.size();
    for (size_t i = 0; i < input_nodes.size(); ++i) {
      VERIFY_EQ(input_nodes[i], input_nodes_[i])
        << "arg_nodes is invalid at index " << i
        << " with value " << input_nodes_[i]
        << ", Expected " << input_nodes[i];
    }
  } else if (version == "cvm_1.1.0") {
    node_row_ptr_ = node_row_ptr;
    input_nodes_ = input_nodes;
  } else {
    LOG(FATAL) << "graph version " << version << " not supported";
  }

  // Check topological order, dependent attribute node_row_ptr
  for (size_t i = 0; i < nodes_.size(); ++i) {
    auto eid = entry_id(i, 0);
    for (auto e: nodes_[i].inputs) {
      VERIFY_LT(entry_id(e), eid)
        << "the graph does not follow the topological order.";
    }
  }

}

void CvmRuntime::Setup() {
  this->PlanStorage();
  this->SetupStorage();
  this->SetupOpExecs();
}

void CvmRuntime::GetShape(int index, DLTensor* t) {
  VERIFY_LE(index, attrs_.shape.size());
  auto shape = attrs_.shape[index];
  t->ndim = shape.size();
  if (t->shape) delete t->shape;
  t->shape = new int64_t[t->ndim];
  for (int i = 0; i < t->ndim; ++i) {
    t->shape[i] = shape[i];
  }
}

void CvmRuntime::GetOutputShape(int index, DLTensor* t) {
  VERIFY_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);
  GetShape(eid, t);
}

/*!
 * \brief Get the input index given the name of input.
 * \param name The name of the input.
 * \return The index of input.
 */
int CvmRuntime::GetInputIndex(const std::string& name) {
  for (size_t i = 0; i< input_nodes_.size(); ++i) {
    uint32_t nid = input_nodes_[i];
    if (nodes_[nid].name() == name) {
      return static_cast<int>(i);
    }
  }
  LOG(FATAL) << "cannot find `" << name << "` among input";
  return -1;
}

void CvmRuntime::SetData(int index, DLTensor* data_in) {
  VERIFY((0 <= index && 
        static_cast<size_t>(index) < input_nodes_.size()))
    << "input index out of range [0, "
    << input_nodes_.size() << "), but " << index;
  VERIFY(nodes_[input_nodes_[index]].is_data())
    << "set input must named `data`";
  
  this->SetInput(index, data_in);
}
/*!
 * \brief set index-th input to the graph.
 * \param index The input index.
 * \param data_in The input data.
 */
void CvmRuntime::SetInput(int index, DLTensor* data_in) {
  VERIFY((0 <= index &&
        static_cast<size_t>(index) < input_nodes_.size()))
    << "input index out of range [0, "
    << input_nodes_.size() << "), but " << index;
  uint32_t nid = input_nodes_[index];
  uint32_t eid = this->entry_id(nid, 0);

  auto dtype = data_in->dtype;
  VERIFY((dtype.code == kDLInt) &&
         (dtype.bits == 32) &&
         (dtype.lanes == 1))
    << "cvm runtime only supported INT32 NDArray, but ("
    << dtype.code << ", " << dtype.bits << ", " << dtype.lanes << ")";
  auto ctx = data_in->ctx;
  VERIFY_EQ(ctx.device_type, kDLCPU)
    << "cvm runtime only supported input with `cpu` device"
    << ", but " << ctx.device_type;

  // Load input data with check
  int ndim = data_in->ndim;
  auto dshp = data_in->shape;
  auto expected = attrs_.shape[eid];
  uint64_t size = 1;
  VERIFY_EQ(ndim, expected.size())
    << "Loaded data shape ndim " << ndim
    << " not matched " << expected.size();
  for (int i = 0; i < ndim; ++i) {
    VERIFY_EQ(dshp[i], expected[i])
      << "Loaded data shape at index " << i
      << " with value " << dshp[i]
      << ", Expected " << expected[i];
    size *= dshp[i];
  }

  // Precision check
  int32_t *data = static_cast<int32_t*>(data_in->data);
  auto& prec = this->attrs_.precision[eid];
  int32_t range = (1 << (prec - 1)) - 1;
  if (nodes_[nid].is_data()) {
    for (uint64_t i = 0; i < size; ++i) {
      if (data[i] > range) data[i] = range;
      else if (data[i] < -range) data[i] = -range;
    }
  } else {
    for (uint64_t i = 0; i < size; ++i) {
      VERIFY(((-range <= data[i]) && (data[i] <= range)))
        << "parameter " << nodes_[nid].name()
        << " at index " << i << " value:" << data[i]
        << " exceed of precision " << prec;
    }
  }

  data_entry_[eid].CopyFrom(data_in);
}
/*!
 * \brief Get the number of outputs
 *
 * \return The number of outputs from graph.
 */
int CvmRuntime::NumOutputs() const {
  return outputs_.size();
}
/*!
 * \brief Return NDArray for given input index.
 * \param index The input index.
 *
 * \return NDArray corresponding to given input node index.
 */
NDArray CvmRuntime::GetInput(int index) const {
  VERIFY_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  return data_entry_[eid];
}
/*!
 * \brief Return NDArray for given output index.
 * \param index The output index.
 *
 * \return NDArray corresponding to given output node index.
 */
NDArray CvmRuntime::GetOutput(int index) const {
  VERIFY_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);
  return data_entry_[eid];
}

int CvmRuntime::GetOutputPrecision() {
  int ret = 0;
  for (unsigned int index = 0; index < outputs_.size(); ++index) {
    uint32_t eid = this->entry_id(outputs_[index]);
    int precision = attrs_.precision[eid];
    ret = std::max(ret, precision);
  }
  return ret;
}

int CvmRuntime::GetInputPrecision() {
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].is_data()) {
      return attrs_.precision[entry_id(i, 0)];
    }
  }
  LOG(FATAL) << "can not find input `data`";
  return -1;
}

int CvmRuntime::GetOutputNum() {
  return static_cast<int>(outputs_.size());
}

/*!
 * \brief Copy index-th output to data_out.
 * \param index The output index.
 * \param data_out the output data.
 */
void CvmRuntime::CopyOutputTo(int index, DLTensor* data_out) {
  VERIFY_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);

  // Check the shapes to avoid receiving in different dimension but same size.
  const NDArray& data = data_entry_[eid];
  VERIFY_EQ(data->ndim, data_out->ndim);
  for (int32_t j = 0; j < data->ndim; ++j) {
    VERIFY_EQ(data->shape[j], data_out->shape[j]);
  }

  data_entry_[eid].CopyTo(data_out);
}

/*!
 * \brief Load parameters from parameter blob.
 * \param param_blob A binary blob of parameter.
 */
void CvmRuntime::LoadParams(const std::string& param_blob) {
  utils::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
  this->LoadParams(&strm);
}

void CvmRuntime::LoadParams(utils::Stream* strm) {
  uint64_t header, reserved;
  VERIFY(strm->Read(&header))
      << "Invalid parameters file format";
  VERIFY(header == kCVMNDArrayListMagic)
      << "Invalid parameters file format";
  VERIFY(strm->Read(&reserved))
      << "Invalid parameters file format";

  std::vector<std::string> names;
  VERIFY(strm->Read(&names))
      << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  VERIFY(size == names.size())
      << "Invalid parameters file format";

  std::vector<bool> set_flag(input_nodes_.size(), false);
  for (size_t i = 0; i < size; ++i) {
    int in_idx = GetInputIndex(names[i]);
    set_flag[in_idx] = true;

    // The data_entry is allocated on device, NDArray.load always load the array into CPU.
    NDArray temp;
    temp.Load(strm);
    this->SetInput(in_idx, const_cast<DLTensor*>(temp.operator->()));
  }

  for (size_t i = 0; i < set_flag.size(); ++i) {
    uint32_t nid = input_nodes_[i];
    VERIFY((set_flag[i] || (nodes_[nid].is_data())))
      << "parameter nid=" << nid
      << " name=" << nodes_[nid].name() << " has not been loaded";
  }
}

void CvmRuntime::PlanStorage() {
  // Grab saved optimization plan from graph.
  std::vector<CVMType> vtype(this->num_node_entries_,
          cvm::runtime::String2CVMType("int32"));

  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    // Use the fallback device if no device index is available.
    int device_type = static_cast<int>(ctxs_[0].device_type);
    if (!attrs_.device_index.empty()) {
      device_type = attrs_.device_index[i];
    }
    auto size = TShape(attrs_.shape[i]).Size();
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    VERIFY(bits % 8U ==  0U || bits ==1U);
    size_t bytes = ((bits + 7U) / 8U) * size;

    uint32_t sid = static_cast<uint32_t>(attrs_.storage_id[i]);
    if (sid >= pool_entry.size()) {
      pool_entry.resize(sid + 1, {0, -1});
    } else {
      VERIFY(pool_entry[sid].device_type == -1 ||
            pool_entry[sid].device_type == device_type)
          << "The same pool entry cannot be assigned to multiple devices";
    }
    pool_entry[sid].size = std::max(pool_entry[sid].size, bytes);
    pool_entry[sid].device_type = device_type;
  }
}

int64_t CvmRuntime::GetStorageSize() {
  int64_t ret = 0;
  int64_t MAX_STORAGE = (int64_t)1<<32;
  for (const auto& pit : pool_entry) {
    ret += (static_cast<int64_t>(pit.size + 3) / 4) * 4;
    VERIFY_LE(ret, MAX_STORAGE)
      << "storage size exceed MAX_STORAGE " << MAX_STORAGE;
  }
  return ret;
}

void CvmRuntime::SetupStorage() {
  // Grab saved optimization plan from graph.
  std::vector<CVMType> vtype(this->num_node_entries_,
      cvm::runtime::String2CVMType("int32"));

  // Allocate the space.
  for (const auto& pit : pool_entry) {
    std::vector<int64_t> shape;
    // This for loop is very fast since there are usually only a couple of
    // devices available on the same hardware.
    const auto& cit =
        std::find_if(ctxs_.begin(), ctxs_.end(), [&pit](const CVMContext& c) {
          return pit.device_type == static_cast<int>(c.device_type);
        });
    CVMContext ctx = cit == ctxs_.end() ? ctxs_[0] : *cit;
    shape.push_back(static_cast<int64_t>(pit.size + 3) / 4);
    storage_pool_.push_back(
        NDArray::Empty(shape, DLDataType{kDLInt, 32, 1}, ctx));
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  data_entry_.resize(num_node_entries());

  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    CHECK_LT(static_cast<size_t>(storage_id), storage_pool_.size());
    data_entry_[i] =
        storage_pool_[storage_id].CreateView(attrs_.shape[i], vtype[i]);
  }

}

void CvmRuntime::SetupOpExecs() {
  op_execs_.resize(this->GetNumOfNodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < this->GetNumOfNodes(); ++nid) {
    auto& inode = nodes_[nid];
    if (inode.is_variable()) continue;
    std::vector<DLTensor> args;
    for (const auto& e : inode.inputs) {
      args.push_back(*(data_entry_[this->entry_id(e)].operator->()));
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(*(data_entry_[eid].operator->()));
    }
    op_execs_[nid] = CreateCVMOp(inode.param, &inode.attrs, args, inode.inputs.size());
  }
}

std::function<void()> CvmRuntime::CreateCVMOp(
    const CVMOpParam& param, NodeAttrs* attr,
    const std::vector<DLTensor>& args,
    size_t num_inputs) {
  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<CVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };
  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = std::move(args);
  // if (param.flatten_data) {
  //   arg_ptr->shape_data.resize(arg_ptr->args.size());
  // }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    CVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    // if (param.flatten_data) {
    //   arg_ptr->shape_data[i] = std::accumulate(
    //       t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
    //   t->ndim = 1;
    //   t->shape = &(arg_ptr->shape_data[i]);
    // }
  }
  CVMValue t_attr;
  t_attr.v_handle = (void*)attr;
  arg_ptr->arg_values.push_back(t_attr);
  arg_ptr->arg_tcodes.push_back(kHandle);

  if (param.func_name == "__nop") {
    return [](){};
  } else if (param.func_name == "__copy") {
    // Perform cross device data copy.
    // Directly copy data from the input to the output.
    auto fexec = [arg_ptr]() {
      DLTensor* from = static_cast<DLTensor*>(arg_ptr->arg_values[0].v_handle);
      DLTensor* to = static_cast<DLTensor*>(arg_ptr->arg_values[1].v_handle);
      CVM_CCALL(CVMArrayCopyFromTo(from, to, nullptr));
    };
    return fexec;
  }

  // Get compiled function from the module that contains both host and device
  // code.
  auto op = param.func_name;
  int device_type = static_cast<int>(ctxs_[0].device_type);
  std::string module_name = "cvm.runtime.cvm";
  if (device_type == kDLGPU) module_name += "_cuda";
  module_name += ".";
  auto func = cvm::runtime::Registry::Get(module_name + op);
  VERIFY(func != nullptr) << "function undefined " << module_name + op;
  return [arg_ptr, op, func](){
    CVMRetValue rv;
    CVMArgs targs(
      arg_ptr->arg_values.data(),
      arg_ptr->arg_tcodes.data(),
      static_cast<int>(arg_ptr->arg_values.size())
    );
    func->CallPacked(targs, &rv);
  };

  return [](){};
}

PackedFunc CvmRuntime::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          this->SetData(in_idx, args[1]);
        } else {
          this->SetData(args[0], args[1]);
        }
        CALL_END();
      });
  } else if (name == "get_output") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->CopyOutputTo(args[0], args[1]);
        CALL_END();
      });
  } else if (name == "get_input_shape") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          if (in_idx >= 0)
            this->GetShape(in_idx, args[1]);
          else
            *rv = -1;
        } else {
          this->GetShape(args[0], args[1]);
        }
        CALL_END();
      });
  } else if (name == "get_storage_size") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kHandle) {
          void *placeholder = args[0];
          VERIFY(placeholder != NULL);
          *static_cast<int64_t*>(placeholder) = this->GetStorageSize();
        } else {
          *rv = -1;
        }
        CALL_END();
      });
  } else if (name == "get_version") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kArrayHandle) {
          void *placeholder = args[0];
          VERIFY(placeholder != NULL);
          strcpy(static_cast<char*>(placeholder), this->version().c_str());
        } else {
          *rv = -1;
        }
        CALL_END();
      });
  } else if (name == "get_postprocess_method") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kStr) {
          char *placeholder = args[0].ptr<char>();
          VERIFY(placeholder != NULL);
          strcpy(static_cast<char*>(placeholder), this->postprocess_method().c_str());
        } else {
          *rv = -1;
        }
        CALL_END();
      });
  } else if (name == "get_ops") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kHandle) {
          void *placeholder = args[0];
          VERIFY(placeholder != NULL);
          *static_cast<int64_t*>(placeholder) = this->GetOps();
        } else {
          *rv = -1;
        }
        CALL_END();
      });
  } else if (name == "get_output_precision") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kHandle) {
          void *placeholder = args[0];
          VERIFY(placeholder != NULL);
          auto precision = this->GetOutputPrecision();
          *static_cast<int32_t*>(placeholder) = precision;
        } else {
          *rv = -1;
        }
        CALL_END();
      });
  } else if (name == "get_input_precision") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kHandle) {
          void *placeholder = args[0];
          VERIFY(placeholder != NULL);
          auto precision = this->GetInputPrecision();
          *static_cast<int32_t*>(placeholder) = precision;
        } else {
          *rv = -1;
        }
        CALL_END();
      });
  } else if (name == "get_output_num") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kHandle) {
          void *placeholder = args[0];
          VERIFY(placeholder != NULL);
          auto num_output = this->GetOutputNum();
          *static_cast<int32_t*>(placeholder) = num_output;
        } else {
          *rv = -1;
        }
        CALL_END();
      });
  } else if (name == "get_output_shape") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->GetOutputShape(args[0], args[1]);
        CALL_END();
      });
  } else if (name == "run") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->Run();
        CALL_END();
      });
  } else if (name == "setup") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->Setup();
        CALL_END();
      });
  } else if (name == "init") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->Init();
        CALL_END();
      });
  } else if (name == "load_params") {
    return PackedFunc([this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->LoadParams(args[0]);
        CALL_END();
      });
  } else {
    return PackedFunc([](CVMArgs args, CVMRetValue *rv) {
        CALL_BEGIN();
        CALL_END();
      });
  }
}

Module CvmRuntimeCreate(const std::string& sym_json, const std::vector<CVMContext>& ctxs) {
  std::shared_ptr<CvmRuntime> exec = std::make_shared<CvmRuntime>();
  exec->SetGraph(sym_json);
  exec->SetContext(ctxs);
  return Module(exec);
}

// Get all context for the host and other runtime devices.
std::vector<CVMContext> CVMGetAllContext(const CVMArgs& args) {
  // Reserve the first item as the fallback device.
  std::vector<CVMContext> ret;
  CVMContext ctx;
  for (int i = 1; i < args.num_args; i += 2) {
    int dev_type = args[i];
    ctx.device_type = static_cast<DLDeviceType>(dev_type);
    ctx.device_id = args[i + 1];
    ret.push_back(ctx);
  }
  return ret;
}

// 4-argument version is currently reserved to keep support of calling
// from cvm4j and javascript, since they don't have heterogeneous
// execution support yet. For heterogenenous execution, at least 5 arguments will
// be passed in. The third one is the number of devices.
// Eventually, we will only probably pass CVMContext for all the languages.
CVM_REGISTER_GLOBAL("cvm.runtime.create")
  .set_body([](CVMArgs args, CVMRetValue* rv) {
    const auto& contexts = CVMGetAllContext(args);
    *rv = CvmRuntimeCreate(args[0], contexts);
  });


CVM_REGISTER_GLOBAL("cvm.runtime.estimate_ops")
  .set_body([](CVMArgs args, CVMRetValue* rv) {
    try {
      VERIFY_GE(args.num_args, 1) << "The expected number of arguments for "
                                    "graph_runtime.estimate_ops is "
                                    "at least 1, but it has "
                                << args.num_args;
      *rv = CvmRuntime::EstimateOps(args[0]);
    } catch (std::runtime_error &e) {
      *rv = -1;
    } catch (std::logic_error &e) {
      *rv = -2;
    }
  });
}  // namespace runtime
}  // namespace cvm
