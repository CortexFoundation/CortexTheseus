/*!
 *  Copyright (c) 2017 by Contributors
 * \file workspace_pool.h
 * \brief Workspace pool utility.
 */
#ifndef CVM_RUNTIME_WORKSPACE_POOL_H_
#define CVM_RUNTIME_WORKSPACE_POOL_H_

#include <cvm/runtime/device_api.h>
#include <vector>
#include <memory>

namespace cvm {
namespace runtime {
/*!
 * \brief A workspace pool to manage
 *
 *  \note We have the following assumption about backend temporal
 *   workspace allocation, and will optimize for such assumption,
 *   some of these assumptions can be enforced by the compiler.
 *
 *  - Only a few allocation will happen, and space will be released after use.
 *  - The release order is usually in reverse order of allocate
 *  - Repeative pattern of same allocations over different runs.
 */
class CVM_DLL WorkspacePool {
 public:
  /*!
   * \brief Create pool with specific device type and device.
   * \param device_type The device type.
   * \param device The device API.
   */
  WorkspacePool(DLDeviceType device_type, std::shared_ptr<DeviceAPI> device);
  /*! \brief destructor */
  ~WorkspacePool();
  /*!
   * \brief Allocate temporal workspace.
   * \param ctx The context of allocation.
   * \param size The size to be allocated.
   */
  void* AllocWorkspace(CVMContext ctx, size_t size);
  /*!
   * \brief Free temporal workspace in backend execution.
   *
   * \param ctx The context of allocation.
   * \param ptr The pointer to be freed.
   */
  void FreeWorkspace(CVMContext ctx, void* ptr);

 private:
  class Pool;
  /*! \brief pool of device local array */
  std::vector<Pool*> array_;
  /*! \brief device type this pool support */
  DLDeviceType device_type_;
  /*! \brief The device API */
  std::shared_ptr<DeviceAPI> device_;
};

}  // namespace runtime
}  // namespace cvm
#endif  // CVM_RUNTIME_WORKSPACE_POOL_H_
