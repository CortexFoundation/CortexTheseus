/*!
 *  Copyright (c) 2016 by Contributors
 * \file cpu_device_api.cc
 */
#include <utils/logging.h>
#include <utils/thread_local.h>
#include <cvm/runtime/registry.h>
#include <cvm/runtime/device_api.h>
#include <cstdlib>
#include <cstring>

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

namespace cvm {
namespace runtime {
class CPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(CVMContext ctx) final {}
  void GetAttr(CVMContext ctx, DeviceAttrKind kind, CVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(CVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       CVMType type_hint) final {
    void* ptr;
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr) throw std::bad_alloc();
#elif defined(_LIBCPP_SGX_CONFIG) || (defined(__ANDROID__) && __ANDROID_API__ < 16)
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr) throw std::bad_alloc();
#else
    // posix_memalign is available in android ndk since __ANDROID_API__ >= 16
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0) throw std::bad_alloc();
#endif
    return ptr;
  }

  void FreeDataSpace(CVMContext ctx, void* ptr) final {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      CVMContext ctx_from,
                      CVMContext ctx_to,
                      CVMType type_hint,
                      CVMStreamHandle stream) final {
    memcpy(static_cast<char*>(to) + to_offset,
           static_cast<const char*>(from) + from_offset,
           size);
  }

  static const std::shared_ptr<CPUDeviceAPI>& Global() {
    static std::shared_ptr<CPUDeviceAPI> inst =
        std::make_shared<CPUDeviceAPI>();
    return inst;
  }
};

CVM_REGISTER_GLOBAL("device_api.cpu")
.set_body([](CVMArgs args, CVMRetValue* rv) {
    DeviceAPI* ptr = CPUDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });
}  // namespace runtime
}  // namespace cvm
