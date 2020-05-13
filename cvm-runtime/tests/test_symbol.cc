#include <cvm/c_api.h>
#include <cvm/model.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <cvm/runtime/registry.h>
#include <cvm/op.h>
#include <cvm/op_attr_types.h>
#include <cvm/runtime/ndarray.h>
#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/serializer.h>
#include <cvm/node.h>
#include <cvm/runtime/c_runtime_api.h>
#include <cvm/c_symbol_api.h>
#include <cvm/symbolic.h>
#include <cvm/c_api_graph.h>
#include <cvm/graph.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

using cvm::runtime::PackedFunc; using cvm::runtime::Registry;
using namespace cvm;
using namespace cvm::runtime;

struct OpInfo{
  const char *name, *description, *return_type;
  nn_uint num_doc_args;
  const char **arg_names, **arg_type_infos, **arg_descriptions;
  void PrintOpInfo(){
    printf("name = %s\n", name);
    printf("description = %s \n", description);
    printf("return_type = %s\n", return_type);
    printf("arg info:\n");
    for(int i = 0; i < num_doc_args; ++i){
      printf("%s %s %s\n", arg_names[i], arg_type_infos[i], arg_descriptions[i]);
    }
  }
};

int main(){
  nn_uint n = 0;
  const char** opNameList;
  int ret = CVMListAllOpNames(&n, &opNameList); 
  if(ret == -1){
    printf("get op names failed.\n");
    return 0;
  }

  //display all op names
  //for(int i = 0; i < n; ++i){
  //  printf("op %d: %s\n", i, opNameList[i]); 
  //}

  printf("get clip handle.\n");
  OpHandle op;
  ret = CVMGetOpHandle("clip", &op);
  if(ret == -1){
    printf("get op handle failed.\n");
    return 0;
  }

  printf("get clip info:\n");
  OpInfo opInfo;
  ret = CVMGetOpInfo(
      op, 
      &opInfo.name, 
      &opInfo.description, 
      &opInfo.num_doc_args,
      &opInfo.arg_names,
      &opInfo.arg_type_infos,
      &opInfo.arg_descriptions,
      &opInfo.return_type);
  opInfo.PrintOpInfo();

  const char *keys[] = {"a_min", "a_max"};
  const char *vals[] = {"0", "10"};
  SymbolHandle symbolHandle;
  nn_uint num_param = 2;
  printf("create clip symble handle.\n");
  ret = CVMSymbolCreateAtomicSymbol(op, num_param, keys, vals, &symbolHandle);
  if(ret != 0){
    printf("create clip symble failed.\n");
    return 0;
  }

  Symbol *clip_symbol = (Symbol*)symbolHandle;
  //clip_symbol.Print(std::cout);
  //vector<cvm::NodePtr> nodePtr = symbol->ListInputs(Symbol::kAll);
  //Symbol xsymbol = Symbol::CreateVariable("x");
  //int a[10] = {0};
  //vector<int> b;
  //cvm::NodePtr np = cvm::Node::Create();
  //
  //
  Symbol x = Symbol::CreateVariable("x");
  std::vector<const Symbol*> vec_args(1);
  vec_args[0] = &x;
  utils::array_view<const Symbol*> args(vec_args);
  const std::string ret_name = "y";
  std::unordered_map<std::string, const Symbol*> kwargs;
  clip_symbol->Compose(args, kwargs, ret_name);

  printf("create graph.\n");
  GraphHandle graph;
  ret = CVMGraphCreate(symbolHandle, &graph);
  if(ret != 0){
    printf("create graph failed.\n");
    return 0;
  }

  const char *key[] = {"shape_inputs", "dtype_inputs", "precision_inputs", "target"};
  const char *json_value[] = {"[\"list_shape\", [[1]]]", "[\"list_int\", [8]]", "[\"list_int\", [8]]", "\"str\", \"cpu\""};
  for(int i = 0; i < 3; i++){
    ret = CVMGraphSetJSONAttr(graph, key[i], json_value[i]);
    if(ret != 0){
      printf("graph set json attr (%d, %s) failed.\n", i, key[i]);
      return 0;
    }
  }
  
  GraphHandle dstGraph;
  // const char* pass_names[] = {"InferShape", "InferType", "InferPrecision", "GraphCompile"};
  const char* pass_names[] = {"InferShape", "GraphCompile", "SaveJSON"};
  ret = CVMGraphApplyPasses(graph, 3, pass_names, &dstGraph);
  if(ret != 0){
    printf("apply pass GraphCompile failed.\n");
    return 0;
  }

  const char* json_str;
  int success;
  ret = CVMGraphGetJSONAttr(dstGraph, "json", &json_str, &success);
  if(ret != 0 || success == 0){
    printf("get json string failed.\n");
    return 0;
  }

  std::ofstream of("./tmp.txt");
  of << json_str;
  of.close();

  CVMGraphFree(graph);
  return 0;
}
