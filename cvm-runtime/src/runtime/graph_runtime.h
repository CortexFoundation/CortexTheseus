/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief Tiny graph runtime that can run graph
 *        containing only cvm PackedFunc.
 * \file graph_runtime.h
 */
#ifndef CVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_
#define CVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_

#include <cvm/dlpack.h>
#include <utils/memory_io.h>
#include <utils/json.h>
#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/node.h>
#include <cvm/top/nn.h>

#include <memory>
#include <utility>
#include <vector>
#include <string>

namespace cvm {
namespace runtime {

using cvm::NodeAttrs;

/*! \brief macro to do C API call */
#define CVM_CCALL(func)                                            \
  {                                                                \
    int ret = (func);                                              \
    CHECK_EQ(ret, 0)                                               \
        << CVMGetLastError();                                      \
  }

/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kCVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*! \brief operator attributes about cvm op */
struct CVMOpParam : public utils::Parameter<CVMOpParam> {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;

  CVMUTIL_DECLARE_PARAMETER(CVMOpParam) {
    CVMUTIL_DECLARE_FIELD(func_name);
    CVMUTIL_DECLARE_FIELD(num_inputs).set_default(1);
    CVMUTIL_DECLARE_FIELD(num_outputs).set_default(1);
    CVMUTIL_DECLARE_FIELD(flatten_data).set_default(0);
  }
};


/*!
 * \brief Tiny graph runtime.
 *
 *  This runtime can be acccesibly in various language via
 *  CVM runtime PackedFunc API.
 */
class CvmRuntime : public ModuleNode {
 public:
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const std::string& name) final;

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final {
    return "CvmRuntime";
  }
  void Run();

  std::string postprocess_method() const {
    return postprocess_method_.substr(0, 32);
  }

  std::string version() const {
    return version_.substr(0, 32);
  }
  /*!
   * \brief Initialize the graph executor with graph and context.
   * \param graph_json The execution graph.
   * \param module The module containing the compiled functions for the host
   *  processor.
   * \param ctxs The context of the host and devices where graph nodes will be
   *  executed on.
   */

  void SetGraph(const std::string& graph_json);
  void SetContext(const std::vector<CVMContext>& ctxs);

  int64_t GetOps();

  static int64_t EstimateOps(const std::string& sym_json) {
    CvmRuntime rt;
    rt.SetGraph(sym_json);
    rt.Init();
    return rt.GetOps();
  }
  /*!
   * \brief Get the input index given the name of input.
   * \param name The name of the input.
   * \return The index of input.
   */
  void GetShape(int index, DLTensor* data);

  void GetOutputShape(int index, DLTensor* data);

  int GetInputIndex(const std::string& name);

  /*!
   * \brief set named data input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetData(int index, DLTensor* data_in);
  /*!
   * \brief set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(int index, DLTensor* data_in);
  /*!
   * \brief Get the number of outputs
   *
   * \return The number of outputs from graph.
   */
  int NumOutputs() const;
  /*!
   * \brief Return NDArray for given input index.
   * \param index The input index.
   *
   * \return NDArray corresponding to given input node index.
   */
  NDArray GetInput(int index) const;
  /*!
   * \brief Return NDArray for given output index.
   * \param index The output index.
   *
   * \return NDArray corresponding to given output node index.
   */
  NDArray GetOutput(int index) const;
  /*!
   * \brief Copy index-th output to data_out.
   * \param index The output index.
   * \param data_out the output data.
   */
  void CopyOutputTo(int index, DLTensor* data_out);
  /*!
   * \brief Load parameters from binary stream
   * \param strm The input stream.
   */
  void LoadParams(utils::Stream* strm);
  /*!
   * \brief Load parameters from parameter blob.
   * \param param_blob A binary blob of parameter.
   */
  void LoadParams(const std::string& param_blob);
 /*!
  * \brief Get total number of nodes.
  * \return Total number of nodes.
  */
  uint32_t GetNumOfNodes() const {
    return static_cast<uint32_t>(nodes_.size());
  }

  std::string GetNodeName(uint32_t nid) const {
    return nodes_[nid].name();
  }


 protected:
  // Memory pool entry.
  struct PoolEntry {
    size_t size;
    int device_type;
    PoolEntry(int s, int dev_type) : size(s), device_type(dev_type) {}
  };
  // Node entry
  struct NodeEntry {
    uint32_t node_id;
    uint32_t index;
    uint32_t version;
    // JSON Loader
    void Load(utils::JSONReader *reader) {
      reader->BeginArray();
      VERIFY(reader->NextArrayItem()) << "invalid json format";
      reader->Read(&node_id);
      VERIFY(reader->NextArrayItem()) << "invalid json format";
      reader->Read(&index);
      if (reader->NextArrayItem()) {
        reader->Read(&version);
        VERIFY(!reader->NextArrayItem()) << "invalid json format";
      } else {
        version = 0;
      }
    }
  };
  // Node
  struct Node {
    // operator type in string
    std::string op_type;
    // parameters
    CVMOpParam param;
    // precision
    int precision;
    // inputs
    std::vector<NodeEntry> inputs;
    // op attr
    NodeAttrs attrs;
    // JSON Loader
    void LoadAttrs(utils::JSONReader *reader, CVMOpParam* param) {
      int bitmask = 0;
      std::string key, value;
      reader->BeginObject();
      while (reader->NextObjectItem(&key)) {
        reader->Read(&value);
        if (key == "func_name") {
          param->func_name = value;
          bitmask |= 1;
        } else if ((key == "num_inputs") ||
                   (key == "num_outputs") ||
                   (key == "op_attrs") ||
                   (key == "flatten_data")) {
        } else {
          LOG(FATAL) << "node attributes do not support key " << key;
        }
      }
      VERIFY_EQ(bitmask, 1) << "invalid format";
    }
    // JSON Loader
    void Load(utils::JSONReader *reader) {
      reader->BeginObject();
      int bitmask = 0;
      std::string key;
      while (reader->NextObjectItem(&key)) {
        if (key == "op") {
          reader->Read(&op_type);
          VERIFY((op_type == "cvm_op") || (op_type == "null"))
            << "CVM executor only supported cvm_op or parameter vs. " << op_type; 
          bitmask |= 1;
        } else if (key == "name") {
          reader->Read(&attrs.name);
          bitmask |= 2;
        } else if (key == "inputs") {
          reader->Read(&inputs);
          bitmask |= 4;
        } else if (key == "attrs") {
          this->LoadAttrs(reader, &param);
        } else if (key == "precision") {
          reader->Read(&precision);
        } else {
          LOG(FATAL) << "node do not support key " << key;
        }
      }
      VERIFY_EQ(bitmask, 1|2|4) << "invalid format";
    }
    std::string GetOpName(std::string name) {
      std::string ret = name;
      for (int i = name.size() - 1; i >= 0; --i) {
        if (name[i] >= '0' && name[i] <= '9') continue;
        else if (name[i] == '_') ret = name.substr(0, i);
        break;
      }
      return ret;
    }

    inline const Op* op() const { return this->attrs.op; }
    inline std::string name() const { return this->attrs.name; }
    inline bool is_variable() const { return this->op_type == "null"; }
    inline bool is_data() const { return this->is_variable() && (this->name()=="data"); }
    inline uint32_t num_inputs() const {
      if (is_variable()) return 1;
      if (this->op()->get_num_inputs == nullptr) {
        auto num = this->op()->num_inputs;
        if (num == cvm::kVarg) num = inputs.size();
        return num;
      } else {
        return this->op()->get_num_inputs(this->attrs);
      }
    }
    inline uint32_t num_outputs() const {
      if (is_variable()) return 1;
      if (this->op()->get_num_outputs == nullptr) {
        return this->op()->num_outputs;
      } else {
        return this->op()->get_num_outputs(this->attrs);
      }
    }

    void LoadOpAndAttrs(std::string json_) {
      if (is_variable()) return ;
      param.func_name = GetOpName(param.func_name);
      attrs.op = cvm::Op::Get(param.func_name);

      std::istringstream is(json_);
      utils::JSONReader reader(&is);
      reader.Read(&attrs.dict);
      if (attrs.op->attr_parser) {
        attrs.op->attr_parser(&attrs);
      } else {
        VERIFY_EQ(attrs.dict.size(), 0)
          << "operator " << this->op()->name
          << " name=" << this->name() << " should not have attributes"
          << ", but " << json_;
      }

      param.num_inputs = this->num_inputs();
      param.num_outputs = this->num_outputs();
      VERIFY_EQ(param.num_inputs, inputs.size())
        << "operator " << param.func_name << " name=" << attrs.name
        << "'s inputs length invaild " << inputs.size() << " vs. " << param.num_inputs;
    }

  };
  struct GraphAttr {
    std::vector<int> storage_id;
    std::vector<int> device_index;
    std::vector<std::string> dltype;
    std::vector<int> precision;
    std::vector<std::string> op_attrs;
    std::vector<std::vector<int64_t> > shape;
    // The graph attribute fields.
    void Load(utils::JSONReader *reader) {
      reader->BeginObject();
      int bitmask = 0;
      std::string key, type;
      while (reader->NextObjectItem(&key)) {
        if (key == "dltype") {
          reader->BeginArray();
          VERIFY(reader->NextArrayItem());
          reader->Read(&type);
          VERIFY_EQ(type, "list_str");
          VERIFY(reader->NextArrayItem());
          reader->Read(&dltype);
          VERIFY(!reader->NextArrayItem());
        } else if (key == "storage_id") {
          reader->BeginArray();
          VERIFY(reader->NextArrayItem());
          reader->Read(&type);
          VERIFY_EQ(type, "list_int");
          VERIFY(reader->NextArrayItem());
          reader->Read(&storage_id);
          VERIFY(!reader->NextArrayItem());
          bitmask |= 2;
        } else if (key == "shape") {
          reader->BeginArray();
          VERIFY(reader->NextArrayItem());
          reader->Read(&type);
          VERIFY_EQ(type, "list_shape");
          VERIFY(reader->NextArrayItem());
          reader->Read(&shape);
          VERIFY(!reader->NextArrayItem());
          bitmask |= 4;
        } else if (key == "device_index") {
          reader->BeginArray();
          VERIFY(reader->NextArrayItem());
          reader->Read(&type);
          VERIFY_EQ(type, "list_int");
          VERIFY(reader->NextArrayItem());
          reader->Read(&device_index);
          VERIFY(!reader->NextArrayItem());
        } else if (key == "precision") {
          reader->BeginArray();
          VERIFY(reader->NextArrayItem());
          reader->Read(&type);
          VERIFY_EQ(type, "list_int");
          VERIFY(reader->NextArrayItem());
          reader->Read(&precision);
          VERIFY(!reader->NextArrayItem());
        } else if (key == "op_attrs") {
          reader->BeginArray();
          VERIFY(reader->NextArrayItem());
          reader->Read(&type);
          VERIFY_EQ(type, "list_str");
          VERIFY(reader->NextArrayItem());
          reader->Read(&op_attrs);
          VERIFY(!reader->NextArrayItem());
          bitmask |= 8;
        } else if ((key == "dtype")) {
          reader->BeginArray();
          VERIFY(reader->NextArrayItem());
          reader->Read(&type);
          if (type == "list_int") {
            VERIFY(reader->NextArrayItem());
            std::vector<int> temp;
            reader->Read(&temp);
          } else if (type == "list_str") {
            VERIFY(reader->NextArrayItem());
            size_t temp;
            reader->Read(&temp);
          } else {
              LOG(FATAL) << "cannot skip graph attr " << key;
          }
          VERIFY(!reader->NextArrayItem());
        } else {
          LOG(FATAL) << "graph attribute " << key << " not supported";
        }
      }
      VERIFY_EQ(bitmask, 2|4|8) << "invalid format";
    }
  };
  // The graph attribute fields.
  void Load(utils::JSONReader *reader) {
    reader->BeginObject();
    int bitmask = 0;
    std::string key;
    while (reader->NextObjectItem(&key)) {
      if (key == "nodes") {
        reader->Read(&nodes_);
        bitmask |= 1;
      } else if (key == "arg_nodes") {
        reader->Read(&input_nodes_);
      } else if (key == "node_row_ptr") {
        reader->Read(&node_row_ptr_);
      } else if (key == "heads") {
        reader->Read(&outputs_);
        bitmask |= 8;
      } else if (key == "attrs") {
        reader->Read(&attrs_);
        bitmask |= 16;
      } else if (key == "version") {
        reader->Read(&version_);
      } else if (key == "postprocess") {
        reader->Read(&postprocess_method_);
      } else {
        LOG(FATAL) << "key " << key << " in json is not supported";
      }
    }
    VERIFY_EQ(bitmask, 1|8|16) << "invalid format";

  }

  void PrepareGraphWithVersion();
  int GetOutputNum();
  int GetOutputPrecision();
  int GetInputPrecision();
public:
  /*! \brief Setup the shape, type, and precision */
  void Init();
  void Setup();
  void SetupShape();
  void SetupType();
  void SetupPrecision();
  bool CheckAttr();
  void PlanStorage();
  /*! \brief Setup the temporal storage */
  void SetupStorage();
  int64_t GetStorageSize();
  /*! \brief Setup the executors. */
  void SetupOpExecs();
  /*!
   * \brief Create an execution function given input.
   * \param attrs The node attributes.
   * \param args The arguments to the functor, including inputs and outputs.
   * \param num_inputs Number of inputs.
   * \return The created executor.
   */
  std::function<void()> CreateCVMOp(const CVMOpParam& param, NodeAttrs* attr,
                                    const std::vector<DLTensor>& args);
  // Get node entry index.
  uint32_t entry_id(uint32_t nid, uint32_t index) const {
    return node_row_ptr_[nid] + index;
  }
  // Get node entry index.
  uint32_t entry_id(const NodeEntry& e) const {
    return entry_id(e.node_id, e.index);
  }
  // Number of node entries.
  uint32_t num_node_entries() const {
    return this->num_node_entries_;
    // return node_row_ptr_.back();
  }
  /*! \brief The graph nodes. */
  std::vector<Node> nodes_;
  uint32_t num_node_entries_;
  /*! \brief The argument nodes. */
  std::vector<uint32_t> input_nodes_;
  /*! \brief Used for quick entry indexing. */
  std::vector<uint32_t> node_row_ptr_;
  /*! \brief Output entries. */
  std::vector<NodeEntry> outputs_;
  /*! \brief Additional graph attributes. */
  GraphAttr attrs_;
  /*! \brief The code module that contains both host and device code. */
  cvm::runtime::Module module_;
  /*! \brief Execution context of all devices including the host. */
  std::vector<CVMContext> ctxs_;
  /*! \brief Common storage pool for all devices. */
  std::vector<NDArray> storage_pool_;

  std::vector<PoolEntry> pool_entry;
  /*! \brief Data entry of each node. */
  std::vector<NDArray> data_entry_;
  /*! \brief Operator on each node. */

  int64_t extra_space_size_;
  NDArray extra_space_;

  std::vector<std::function<void()> > op_execs_;

  std::string graph_json_;

  std::string version_{std::string("cvm_1.0.0")};

  std::string postprocess_method_;
  std::map<std::string, double> times;
};

std::vector<CVMContext> CVMGetAllContext(const CVMArgs& args);

}  // namespace runtime
}  // namespace cvm

#endif  // CVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_
