/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file cvm/compiler/packed_func_ext.h
 * \brief Extension to enable packed functionn for cvm types
 */
#ifndef CVM_COMPILER_PACKED_FUNC_EXT_H_
#define CVM_COMPILER_PACKED_FUNC_EXT_H_

#include <cvm/runtime/packed_func.h>
#include <cvm/runtime/registry.h>
#include <cvm/graph.h>
#include <cvm/symbolic.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace cvm {
namespace compiler {

using cvm::runtime::PackedFunc;

using AttrDict = std::unordered_map<std::string, std::string>;

/*!
 * \brief Get PackedFunction from global registry and
 *  report error if it does not exist
 * \param name The name of the function.
 * \return The created PackedFunc.
 */
inline const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = cvm::runtime::Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}
}  // namespace compiler
}  // namespace cvm

// Enable the graph and symbol object exchange.
namespace cvm {
namespace runtime {

template<>
struct extension_type_info<cvm::Symbol> {
  static const int code = 16;
};

template<>
struct extension_type_info<cvm::Graph> {
  static const int code = 17;
};

template<>
struct extension_type_info<cvm::compiler::AttrDict> {
  static const int code = 18;
};

}  // namespace runtime
}  // namespace cvm
#endif  // CVM_COMPILER_PACKED_FUNC_EXT_H_
