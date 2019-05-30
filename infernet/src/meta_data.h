/*!
 *  Copyright (c) 2017 by Contributors
 * \file meta_data.h
 * \brief Meta data related utilities
 */
#ifndef CVM_RUNTIME_META_DATA_H_
#define CVM_RUNTIME_META_DATA_H_

#include <utils/json.h>
#include <utils/io.h>
#include <cvm/runtime/packed_func.h>
#include <string>
#include <vector>
#include "runtime_base.h"

namespace cvm {
namespace runtime {

/*! \brief function information needed by device */
struct FunctionInfo {
  std::string name;
  std::vector<CVMType> arg_types;
  std::vector<std::string> thread_axis_tags;

  void Save(utils::JSONWriter *writer) const;
  void Load(utils::JSONReader *reader);
  void Save(utils::Stream *writer) const;
  bool Load(utils::Stream *reader);
};
}  // namespace runtime
}  // namespace cvm

namespace utils {
CVMUTIL_DECLARE_TRAITS(has_saveload, ::cvm::runtime::FunctionInfo, true);
}  // namespace utils
#endif  // CVM_RUNTIME_META_DATA_H_
