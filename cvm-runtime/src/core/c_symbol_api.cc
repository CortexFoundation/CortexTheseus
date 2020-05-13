#include <cvm/c_symbol_api.h>
#include <cvm/op.h>
#include <cvm/errors.h>
#include <cvm/symbolic.h>
#include <cvm/runtime/registry.h>
#include <cvm/op.h>
#include <utils/registry.h>

using namespace cvm;


int CVMListAllOpNames(nn_uint *out_size,
                     const char*** out_array) {
  API_BEGIN();
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();
  ret->ret_vec_str = utils::Registry<Op>::ListAllNames();
  ret->ret_vec_charp.resize(0);
  ret->ret_vec_charp.reserve(ret->ret_vec_str.size());
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_array = utils::BeginPtr(ret->ret_vec_charp);
  *out_size = static_cast<nn_uint>(ret->ret_vec_str.size());
  API_END();
}

int CVMGetOpHandle(const char* op_name,
                  OpHandle* op_out) {
  API_BEGIN();
  *op_out = (OpHandle)Op::Get(op_name);  // NOLINT(*)
  API_END();
}

int CVMListUniqueOps(nn_uint *out_size,
                    OpHandle **out_array) {
  API_BEGIN();
  auto &vec = utils::Registry<Op>::List();
  *out_size = static_cast<nn_uint>(vec.size());
  *out_array = (OpHandle*)(utils::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int CVMAddControlDeps(SymbolHandle handle,
                     SymbolHandle src_dep) {
  API_BEGIN();
  static_cast<Symbol*>(handle)->AddControlDeps(
      *static_cast<Symbol*>(src_dep));
  API_END();
}

int CVMGetOpInfo(OpHandle handle,
                const char **name,
                const char **description,
                nn_uint *num_doc_args,
                const char ***arg_names,
                const char ***arg_type_infos,
                const char ***arg_descriptions,
                const char **return_type) {
  const Op *op = static_cast<const Op *>(handle);
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();

  API_BEGIN();
  *name = op->name.c_str();
  *description = op->description.c_str();
  *num_doc_args = static_cast<nn_uint>(op->arguments.size());
  if (return_type) *return_type = nullptr;
  ret->ret_vec_charp.resize(0);
  ret->ret_vec_charp.reserve(op->arguments.size() * 3);
  for (size_t i = 0; i < op->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(op->arguments[i].name.c_str());
  }
  for (size_t i = 0; i < op->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(op->arguments[i].type_info_str.c_str());
  }
  for (size_t i = 0; i < op->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(op->arguments[i].description.c_str());
  }
  *arg_names = utils::BeginPtr(ret->ret_vec_charp);
  *arg_type_infos = utils::BeginPtr(ret->ret_vec_charp) + op->arguments.size();
  *arg_descriptions = utils::BeginPtr(ret->ret_vec_charp) + (op->arguments.size() * 2);
  API_END();
}

int CVMSymbolCreateAtomicSymbol(OpHandle creator,
                               nn_uint num_param,
                               const char **keys,
                               const char **vals,
                               SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  const Op* op = static_cast<const Op*>(creator);
  std::unordered_map<std::string, std::string> kwargs;
  for (nn_uint i = 0; i < num_param; ++i) {
    kwargs.insert({std::string(keys[i]), std::string(vals[i])});
  }
  *s = Symbol::CreateFunctor(op, std::move(kwargs));
  *out = s;
  API_END_HANDLE_ERROR(delete s;);
}

int CVMSymbolCreateVariable(const char *name, SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = Symbol::CreateVariable(name);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int CVMSymbolCreateGroup(nn_uint num_symbols,
                        SymbolHandle *symbols,
                        SymbolHandle *out) {
  Symbol *s = new Symbol();
  Symbol **sym_arr = (Symbol**)symbols; // NOLINT(*)
  API_BEGIN();
  std::vector<Symbol> syms;
  for (nn_uint i = 0; i < num_symbols; ++i) {
    syms.push_back(*sym_arr[i]);
  }
  *s = Symbol::CreateGroup(syms);
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int CVMSymbolGetOutput(SymbolHandle symbol,
                      nn_uint index,
                      SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = (*static_cast<Symbol*>(symbol))[index];
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int CVMSymbolGetInternals(SymbolHandle symbol,
                         SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = static_cast<Symbol*>(symbol)->GetInternals();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int CVMSymbolGetChildren(SymbolHandle symbol,
                        SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = static_cast<Symbol*>(symbol)->GetChildren();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int CVMSymbolFree(SymbolHandle symbol) {
  API_BEGIN();
  delete static_cast<Symbol*>(symbol);
  API_END();
}

int CVMSymbolCopy(SymbolHandle symbol, SymbolHandle *out) {
  Symbol *s = new Symbol();
  API_BEGIN();
  *s = static_cast<const Symbol*>(symbol)->Copy();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int CVMSymbolPrint(SymbolHandle symbol, const char **out_str) {
  Symbol *s = static_cast<Symbol*>(symbol);
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();
  API_BEGIN();
  std::ostringstream os;
  s->Print(os);
  ret->ret_str = os.str();
  *out_str = (ret->ret_str).c_str();
  API_END();
}

int CVMSymbolGetAttr(SymbolHandle symbol,
                    const char* key,
                    const char** out,
                    int* success) {
  Symbol *s = static_cast<Symbol*>(symbol);
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();
  API_BEGIN();
  if (s->GetAttr(key, &(ret->ret_str))) {
    *out = (ret->ret_str).c_str();
    *success = 1;
  } else {
    *out = nullptr;
    *success = 0;
  }
  API_END();
}

int CVMSymbolSetAttrs(SymbolHandle symbol,
                     nn_uint num_param,
                     const char** keys,
                     const char** vals) {
  Symbol *s = static_cast<Symbol*>(symbol);
  API_BEGIN();
  std::vector<std::pair<std::string, std::string> > kwargs;
  for (nn_uint i = 0; i < num_param; ++i) {
    kwargs.emplace_back(
        std::make_pair(std::string(keys[i]), std::string(vals[i])));
  }
  s->SetAttrs(kwargs);
  API_END();
}

int CVMSymbolListAttrs(SymbolHandle symbol,
                      int option,
                      nn_uint *out_size,
                      const char*** out) {
  Symbol *s = static_cast<Symbol*>(symbol);
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();
  API_BEGIN();
  std::unordered_map<std::string, std::string> attr =
      s->ListAttrs(static_cast<Symbol::ListAttrOption>(option));  // NOLINT(*)

  std::vector<std::string>& attr_list = ret->ret_vec_str;
  attr_list.resize(0);
  attr_list.reserve(attr.size());
  for (const auto& kv : attr) {
    attr_list.push_back(kv.first);
    attr_list.push_back(kv.second);
  }
  *out_size = attr.size();
  ret->ret_vec_charp.clear();
  ret->ret_vec_charp.reserve(ret->ret_vec_str.size());
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out = utils::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int CVMSymbolListInputVariables(SymbolHandle symbol,
                               int option,
                               nn_uint *out_size,
                               SymbolHandle** out_sym_array) {
  Symbol *s = static_cast<Symbol*>(symbol);
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();
  API_BEGIN();
  std::vector<NodePtr> vs = s->ListInputs(Symbol::ListInputOption(option));
  ret->ret_handles.resize(0);
  ret->ret_handles.reserve(vs.size());
  for (size_t i = 0; i < vs.size(); ++i) {
    cvm::Symbol* rs = new cvm::Symbol();
    rs->outputs.push_back(NodeEntry{vs[i], 0, 0});
    ret->ret_handles.push_back(rs);
  }
  *out_size = static_cast<nn_uint>(vs.size());
  *out_sym_array = utils::BeginPtr(ret->ret_handles);
  API_END();
}

int CVMSymbolListInputNames(SymbolHandle symbol,
                           int option,
                           nn_uint *out_size,
                           const char ***out_str_array) {
  Symbol *s = static_cast<Symbol*>(symbol);
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str =
      s->ListInputNames(Symbol::ListInputOption(option));
  ret->ret_vec_charp.resize(0);
  ret->ret_vec_charp.reserve(ret->ret_vec_str.size());
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_size = static_cast<nn_uint>(ret->ret_vec_charp.size());
  *out_str_array = utils::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int CVMSymbolListOutputNames(SymbolHandle symbol,
                            nn_uint *out_size,
                            const char ***out_str_array) {
  Symbol *s = static_cast<Symbol*>(symbol);
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();
  API_BEGIN();
  ret->ret_vec_str = s->ListOutputNames();
  ret->ret_vec_charp.resize(0);
  ret->ret_vec_charp.reserve(ret->ret_vec_str.size());
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_size = static_cast<nn_uint>(ret->ret_vec_charp.size());
  *out_str_array = utils::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int CVMSymbolGetNumOutputs(SymbolHandle symbol,
                           nn_uint *output_count) {
  Symbol *s = static_cast<Symbol*>(symbol);
  API_BEGIN();
  *output_count = static_cast<nn_uint>(s->outputs.size());
  API_END();
}

int CVMSymbolCompose(SymbolHandle sym,
                    const char *name,
                    nn_uint num_args,
                    const char** keys,
                    SymbolHandle* args) {
  API_BEGIN();
  CVMAPIThreadLocalEntry *ret = CVMAPIThreadLocalStore::Get();
  std::string& s_name = ret->ret_str;
  std::unordered_map<std::string, const Symbol*>& kwargs
      = ret->kwarg_symbol;
  kwargs.clear();
  if (name != nullptr) {
    s_name = name;
  } else {
    s_name.clear();
  }
  Symbol* s = static_cast<Symbol*>(sym);
  if (keys == nullptr && num_args != 0) {
    kwargs.clear();
    array_view<const Symbol*> parg(
        (Symbol**)args, (Symbol**)args + num_args); // NOLINT(*)
    s->Compose(parg, kwargs, s_name);
  } else {
    for (nn_uint i = 0; i < num_args; ++i) {
      kwargs[keys[i]] = (Symbol*)args[i];  //  NOLINT(*)
    }
    s->Compose(array_view<const Symbol*>(), kwargs, s_name);
  }
  API_END();
}
