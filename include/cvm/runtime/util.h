/*!
 *  Copyright (c) 2017 by Contributors
 * \file cvm/runtime/util.h
 * \brief Useful runtime util.
 */
#ifndef CVM_RUNTIME_UTIL_H_
#define CVM_RUNTIME_UTIL_H_

#include "c_runtime_api.h"

namespace cvm {
namespace runtime {

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes in the type.
 */
inline bool TypeMatch(CVMType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}
}  // namespace runtime
}  // namespace cvm
// Forward declare the intrinsic id we need
// in structure fetch to enable stackvm in runtime
namespace cvm {
namespace ir {
namespace intrinsic {
/*! \brief The kind of structure field info used in intrinsic */
enum CVMStructFieldKind : int {
  // array head address
  kArrAddr,
  kArrData,
  kArrShape,
  kArrStrides,
  kArrNDim,
  kArrTypeCode,
  kArrTypeBits,
  kArrTypeLanes,
  kArrByteOffset,
  kArrDeviceId,
  kArrDeviceType,
  kArrKindBound_,
  // CVMValue field
  kCVMValueContent,
  kCVMValueKindBound_
};
}  // namespace intrinsic
}  // namespace ir
}  // namespace cvm
#endif  // CVM_RUNTIME_UTIL_H_
