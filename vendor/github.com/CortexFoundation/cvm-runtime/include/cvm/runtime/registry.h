/*!
 *  Copyright (c) 2017 by Contributors
 * \file cvm/runtime/registry.h
 * \brief This file defines the CVM global function registry.
 *
 *  The registered functions will be made available to front-end
 *  as well as backend users.
 *
 *  The registry stores type-erased functions.
 *  Each registered function is automatically exposed
 *  to front-end language(e.g. python).
 *
 *  Front-end can also pass callbacks as PackedFunc, or register
 *  then into the same global registry in C++.
 *  The goal is to mix the front-end language and the CVM back-end.
 *
 * \code
 *   // register the function as MyAPIFuncName
 *   CVM_REGISTER_GLOBAL(MyAPIFuncName)
 *   .set_body([](CVMArgs args, CVMRetValue* rv) {
 *     // my code.
 *   });
 * \endcode
 */
#ifndef CVM_RUNTIME_REGISTRY_H_
#define CVM_RUNTIME_REGISTRY_H_

#include <string>
#include <vector>
#include "packed_func.h"

namespace cvm {
namespace runtime {

/*! \brief Registry for global function */
class Registry {
 public:
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  CVM_DLL Registry& set_body(PackedFunc f);  // NOLINT(*)
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  Registry& set_body(PackedFunc::FType f) {  // NOLINT(*)
    return set_body(PackedFunc(f));
  }
  /*!
   * \brief set the body of the function to be TypedPackedFunc.
   *
   * \code
   *
   * CVM_REGISTER_API("addone")
   * .set_body_typed<int(int)>([](int x) { return x + 1; });
   *
   * \endcode
   *
   * \param f The body of the function.
   * \tparam FType the signature of the function.
   * \tparam FLambda The type of f.
   */
  template<typename FType, typename FLambda>
  Registry& set_body_typed(FLambda f) {
    return set_body(TypedPackedFunc<FType>(f).packed());
  }
  /*!
   * \brief Register a function with given name
   * \param name The name of the function.
   * \param override Whether allow oveeride existing function.
   * \return Reference to theregistry.
   */
  CVM_DLL static Registry& Register(const std::string& name, bool override = false);  // NOLINT(*)
  /*!
   * \brief Erase global function from registry, if exist.
   * \param name The name of the function.
   * \return Whether function exist.
   */
  CVM_DLL static bool Remove(const std::string& name);
  /*!
   * \brief Get the global function by name.
   * \param name The name of the function.
   * \return pointer to the registered function,
   *   nullptr if it does not exist.
   */
  CVM_DLL static const PackedFunc* Get(const std::string& name);  // NOLINT(*)
  /*!
   * \brief Get the names of currently registered global function.
   * \return The names
   */
  CVM_DLL static std::vector<std::string> ListNames();

  // Internal class.
  struct Manager;

 protected:
  /*! \brief name of the function */
  std::string name_;
  /*! \brief internal packed function */
  PackedFunc func_;
  friend struct Manager;
};

/*! \brief helper macro to supress unused warning */
#if defined(__GNUC__)
#define CVM_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define CVM_ATTRIBUTE_UNUSED
#endif

#define CVM_STR_CONCAT_(__x, __y) __x##__y
#define CVM_STR_CONCAT(__x, __y) CVM_STR_CONCAT_(__x, __y)

#define CVM_FUNC_REG_VAR_DEF                                            \
  static CVM_ATTRIBUTE_UNUSED ::cvm::runtime::Registry& __mk_ ## CVM

#define CVM_TYPE_REG_VAR_DEF                                            \
  static CVM_ATTRIBUTE_UNUSED ::cvm::runtime::ExtTypeVTable* __mk_ ## CVMT

/*!
 * \brief Register a function globally.
 * \code
 *   CVM_REGISTER_GLOBAL("MyPrint")
 *   .set_body([](CVMArgs args, CVMRetValue* rv) {
 *   });
 * \endcode
 */
#define CVM_REGISTER_GLOBAL(OpName)                              \
  CVM_STR_CONCAT(CVM_FUNC_REG_VAR_DEF, __COUNTER__) =            \
      ::cvm::runtime::Registry::Register(OpName)

/*!
 * \brief Macro to register extension type.
 *  This must be registered in a cc file
 *  after the trait extension_type_info is defined.
 */
#define CVM_REGISTER_EXT_TYPE(T)                                 \
  CVM_STR_CONCAT(CVM_TYPE_REG_VAR_DEF, __COUNTER__) =            \
      ::cvm::runtime::ExtTypeVTable::Register_<T>()

}  // namespace runtime
}  // namespace cvm
#endif  // CVM_RUNTIME_REGISTRY_H_
