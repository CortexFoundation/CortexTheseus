#include <cvm/runtime/device_api.h>
#include <cvm/runtime/registry.h>
#include <mutex>

namespace cvm {
namespace runtime {

class DeviceAPIManager {
 public:
  static const int kMaxDeviceAPI = 32;
  // Get API
  static DeviceAPI* Get(const CVMContext& ctx) {
    return Get(ctx.device_type);
  }
  static DeviceAPI* Get(int dev_type, bool allow_missing = false) {
    return Global()->GetAPI(dev_type, allow_missing);
  }

 private:
  std::array<DeviceAPI*, kMaxDeviceAPI> api_;
  DeviceAPI* rpc_api_{nullptr};
  std::mutex mutex_;
  // constructor
  DeviceAPIManager() {
    std::fill(api_.begin(), api_.end(), nullptr);
  }
  // Global static variable.
  static DeviceAPIManager* Global() {
    static DeviceAPIManager inst;
    return &inst;
  }
  // Get or initialize API.
  DeviceAPI* GetAPI(int type, bool allow_missing) {
    if (type < kRPCSessMask) {
      if (api_[type] != nullptr) return api_[type];
      std::lock_guard<std::mutex> lock(mutex_);
      if (api_[type] != nullptr) return api_[type];
      api_[type] = GetAPI(DeviceName(type), allow_missing);
      return api_[type];
    } else {
      if (rpc_api_ != nullptr) return rpc_api_;
      std::lock_guard<std::mutex> lock(mutex_);
      if (rpc_api_ != nullptr) return rpc_api_;
      rpc_api_ = GetAPI("rpc", allow_missing);
      return rpc_api_;
    }
  }
  DeviceAPI* GetAPI(const std::string name, bool allow_missing) {
    std::string factory = "device_api." + name;
    auto* f = Registry::Get(factory);
    if (f == nullptr) {
      CHECK(allow_missing)
          << "Device API " << name << " is not enabled.";
      return nullptr;
    }
    void* ptr = (*f)();
    return static_cast<DeviceAPI*>(ptr);
  }
};

DeviceAPI* DeviceAPI::Get(CVMContext ctx, bool allow_missing) {
  return DeviceAPIManager::Get(
      static_cast<int>(ctx.device_type), allow_missing);
}


}  // namespace runtime
}  // namespace cvm

using namespace cvm::runtime;

// set device api
CVM_REGISTER_GLOBAL("__cvm_set_device")
.set_body([](CVMArgs args, CVMRetValue *ret) {
    CVMContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(args[0].operator int());
    ctx.device_id = args[1];
    cvm::runtime::DeviceAPIManager::Get(ctx)->SetDevice(ctx);
  });

// set device api
CVM_REGISTER_GLOBAL("_GetDeviceAttr")
.set_body([](CVMArgs args, CVMRetValue *ret) {
    CVMContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(args[0].operator int());
    ctx.device_id = args[1];

    DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[2].operator int());
    if (kind == kExist) {
      DeviceAPI* api = DeviceAPIManager::Get(ctx.device_type, true);
      if (api != nullptr) {
        api->GetAttr(ctx, kind, ret);
      } else {
        *ret = 0;
      }
    } else {
      DeviceAPIManager::Get(ctx)->GetAttr(ctx, kind, ret);
    }
  });
