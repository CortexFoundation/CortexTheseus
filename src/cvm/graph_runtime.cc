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
  this->CheckAttr();
  this->PlanStorage();
}

void CvmRuntime::Setup() {
  this->SetupStorage();
  this->SetupOpExecs();
}

int64_t CvmRuntime::GetOps(const std::string& graph_json) {
  std::istringstream is(graph_json);
  utils::JSONReader reader(&is);
  this->Load(&reader);
  this->CheckAttr();
  return this->GetOps();
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
    if (nodes_[nid].name == name) {
      return static_cast<int>(i);
    }
  }
  LOG(WARNING) << "Warning: cannot find \"" << name << "\" among input";
  return -1;
}
/*!
 * \brief set index-th input to the graph.
 * \param index The input index.
 * \param data_in The input data.
 */
void CvmRuntime::SetInput(int index, DLTensor* data_in) {
  VERIFY_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
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
  for (size_t i = 0; i < size; ++i) {
    int in_idx = GetInputIndex(names[i]);
    VERIFY_GE(in_idx, 0) << "Found param for non-existent input: " << names[i];
    uint32_t eid = this->entry_id(input_nodes_[in_idx], 0);
    VERIFY_LT(eid, data_entry_.size());

    // The data_entry is allocated on device, NDArray.load always load the array into CPU.
    NDArray temp;
    temp.Load(strm);
    data_entry_[eid].CopyFrom(temp);
  }
}

void CvmRuntime::PlanStorage() {
  // Grab saved optimization plan from graph.
  std::vector<CVMType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    vtype.push_back(cvm::runtime::String2CVMType(s_type));
  }

  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    // Use the fallback device if no device index is available.
    int device_type = static_cast<int>(ctxs_[0].device_type);
    if (!attrs_.device_index.empty()) {
      device_type = attrs_.device_index[i];
    }
    size_t size = 1;
    int len = 0;
    for (int64_t sz : attrs_.shape[i]) {
      VERIFY_LE(sz, 0x7fffffffll);
      len += 32 - __builtin_clz(static_cast<unsigned>(sz));
      size *= static_cast<size_t>(sz);
    }
    VERIFY_LE(len, 48);
    VERIFY_GE(storage_id, 0) << "Do not support runtime shape op";
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    VERIFY(bits % 8U ==  0U || bits ==1U);
    size_t bytes = ((bits + 7U) / 8U) * size;

    uint32_t sid = static_cast<uint32_t>(storage_id);
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
  for (const auto& pit : pool_entry) {
    ret += (static_cast<int64_t>(pit.size + 3) / 4) * 4;
    VERIFY_LE(ret, 0x0000ffffffffffffull);
  }
  return ret;
}

void CvmRuntime::SetupStorage() {
  // Grab saved optimization plan from graph.
  std::vector<CVMType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    vtype.push_back(cvm::runtime::String2CVMType(s_type));
  }

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
    if (inode.op_type == "null") continue;
    std::vector<DLTensor> args;
    for (const auto& e : inode.inputs) {
      args.push_back(*(data_entry_[this->entry_id(e)].operator->()));
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(*(data_entry_[eid].operator->()));
    }
    VERIFY(inode.op_type == "cvm_op") << "Can only take cvm_op as op";
    // std::cerr << inode.name << "\n";
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
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    CVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (param.flatten_data) {
      arg_ptr->shape_data[i] = std::accumulate(
          t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
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
  return [arg_ptr, op, func, device_type](){
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
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          if (in_idx >= 0) this->SetInput(in_idx, args[1]);
        } else {
          this->SetInput(args[0], args[1]);
        }
        CALL_END();
      });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->CopyOutputTo(args[0], args[1]);
        CALL_END();
      });
  } else if (name == "get_input_shape") {
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
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
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
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
  } else if (name == "get_ops") {
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
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
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
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
  } else if (name == "get_output_num") {
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
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
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->GetOutputShape(args[0], args[1]);
        CALL_END();
      });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->Run();
        CALL_END();
      });
  } else if (name == "setup") {
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->Setup();
        CALL_END();
      });
  } else if (name == "init") {
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->Init();
        CALL_END();
      });
  } else if (name == "load_params") {
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue* rv) {
        CALL_BEGIN();
        this->LoadParams(args[0]);
        CALL_END();
      });
  } else {
    return PackedFunc([sptr_to_self, this](CVMArgs args, CVMRetValue *rv) {
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
