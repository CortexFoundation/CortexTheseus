#include <cvm/c_api.h>
#include <cvm/op.h>
#include <cvm/symbolic.h>
#include <cvm/graph.h>
#include <cvm/errors.h>
#include <cvm/pass.h>
#include <cvm/c_api_graph.h>

using namespace cvm;

int CVMGraphCreate(SymbolHandle symbol, GraphHandle *graph) {
  Graph* g = new Graph();
  API_BEGIN();
  g->outputs = static_cast<Symbol*>(symbol)->outputs;
  *graph = g;
  API_END_HANDLE_ERROR(delete g);
}

int CVMGraphFree(GraphHandle handle) {
  API_BEGIN();
  delete static_cast<Graph*>(handle);
  API_END();
}

int CVMGraphGetSymbol(GraphHandle graph, SymbolHandle *symbol) {
  Symbol* s = new Symbol();
  API_BEGIN();
  s->outputs = static_cast<Graph*>(graph)->outputs;
  *symbol = s;
  API_END_HANDLE_ERROR(delete s);
}

int CVMGraphSetNodeEntryListAttr_(GraphHandle handle,
                                 const char* key,
                                 SymbolHandle list) {
  API_BEGIN();
  Symbol* s = static_cast<Symbol*>(list);
  Graph* g = static_cast<Graph*>(handle);
  g->attrs[std::string(key)]
      = std::make_shared<any>(s->outputs);
  API_END();
}

int CVMGraphSetJSONAttr(GraphHandle handle,
                       const char* key,
                       const char* json_value) {
  API_BEGIN();
  Graph* g = static_cast<Graph*>(handle);
  std::string temp(json_value);
  std::istringstream is(temp);
  utils::JSONReader reader(&is);
  cvm::any value;
  reader.Read(&value);
  g->attrs[std::string(key)] = std::make_shared<any>(std::move(value));
  API_END();
}

int CVMGraphGetJSONAttr(GraphHandle handle,
                      const char* key,
                      const char** json_out,
                      int *success) {
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();
  API_BEGIN();
  Graph* g = static_cast<Graph*>(handle);
  std::string skey(key);
  auto it = g->attrs.find(skey);
  if (it != g->attrs.end()) {
    std::ostringstream os;
    utils::JSONWriter writer(&os);
    writer.Write(*it->second.get());
    ret->ret_str = os.str();
    *json_out = (ret->ret_str).c_str();
    *success = 1;
  } else {
    *success = 0;
  }
  API_END();
}

int CVMGraphApplyPasses(GraphHandle src,
                       nn_uint num_pass,
                       const char** pass_names,
                       GraphHandle *dst) {
  Graph* g = new Graph();
  API_BEGIN();
  std::vector<std::string> vpass;
  for (nn_uint i = 0; i < num_pass; ++i) {
    vpass.emplace_back(std::string(pass_names[i]));
  }
  *g = ApplyPasses(*static_cast<Graph*>(src), vpass);
  *dst = g;
  API_END_HANDLE_ERROR(delete g);
}
