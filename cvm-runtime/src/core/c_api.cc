#include <cvm/c_api.h>
#include <cvm/model.h>
#include <cvm/dlpack.h>
#include <cvm/time.h>
#include <cvm/runtime/device_api.h>
#include <cvm/runtime/registry.h>
#include <string.h>

#define CHECK_NOT_NULL(x) CHECK(x != nullptr)
#define CHECK_2_NOT_NULL(x, y) CHECK_NOT_NULL(x); CHECK_NOT_NULL(y);
#define CHECK_3_NOT_NULL(x, y, z) CHECK_2_NOT_NULL(x, y); CHECK_NOT_NULL(z);

using cvm::runtime::CVMModel;

static 
std::unordered_map<int, DLDeviceType> const APIDevTypeMap = {
  {0, kDLCPU},
  {1, kDLGPU},
  {2, kDLFORMAL},
};

int CVMAPILoadModel(const char *graph_json, int graph_strlen,
                    const char *param_bytes, int param_strlen,
                    void **net,
                    int device_type, int device_id) {
  API_BEGIN();
  string graph(graph_json, graph_strlen);
  string params(param_bytes, param_strlen);
  DLContext ctx;
  CHECK(APIDevTypeMap.find(device_type) != APIDevTypeMap.end())
    << "Invalid device type: " << device_type
    << ", only supported 0(CPU), 1(GPU), 2(FORMAL)\n";
  ctx.device_type = APIDevTypeMap.at(device_type);
  // ctx.device_type = (device_type == 0) ? kDLCPU : kDLGPU;
  ctx.device_id = device_id;
  CVMModel *model = new CVMModel(graph, ctx);
  if (!model->IsReady() || model->LoadParams(params)) {
    delete model;
    return ERROR_LOGIC;
  }
  *net = (void *)model;
  API_END();
}


int CVMAPIFreeModel(void *net) {
  API_BEGIN();
  CVMModel* model = static_cast<CVMModel*>(net);
  if (net) delete model;
  API_END();
}

int CVMAPIInference(void *net,
                    char *input_data, int input_len,
                    char *output_data) {
  API_BEGIN();
  CHECK_3_NOT_NULL(net, input_data, output_data);

  TIME_INIT(0);
  CVMModel* model = static_cast<CVMModel*>(net);

  DLTensor *input = model->PlanInput(input_data, input_len);
  auto outputs = model->PlanOutput();
  TIME_ELAPSED(0, "plan input & output");

  TIME_INIT(1);
  model->Run(input, outputs);
  TIME_ELAPSED(1, "model operators inference");

  TIME_INIT(2);
  model->SaveTensor(outputs, output_data);
  if (input) CVMArrayFree(input);
  for (auto &output : outputs) CVMArrayFree(output);
  TIME_ELAPSED(2, "temporary variable free");

  API_END();
}

int CVMAPIGetVersion(void *net, char *version) {
  API_BEGIN();
  CHECK_2_NOT_NULL(net, version);

  CVMModel* model = static_cast<CVMModel*>(net);
  strcpy(version, model->GetVersion().c_str());
  API_END();
}

int CVMAPIGetPreprocessMethod(void *net, char *method) {
  API_BEGIN();
  CHECK_2_NOT_NULL(net, method);

  CVMModel* model = static_cast<CVMModel*>(net);
  strcpy(method, model->GetPostprocessMethod().c_str());
  API_END();
}

int CVMAPIGetInputLength(void *net, unsigned long long *size) {
  API_BEGIN();
  CHECK_2_NOT_NULL(net, size);
  CVMModel* model = static_cast<CVMModel*>(net);
  *size = static_cast<unsigned long long>(model->GetInputLength());
  API_END();
}

int CVMAPIGetOutputLength(void *net, unsigned long long *size) {
  API_BEGIN();
  CHECK_2_NOT_NULL(net, size);
  CVMModel* model = static_cast<CVMModel*>(net);
  *size = static_cast<unsigned long long>(model->GetOutputLength());
  API_END();
}

int CVMAPIGetOutputTypeSize(void *net, unsigned long long *size) {
  API_BEGIN();
  CHECK_2_NOT_NULL(net, size);
  CVMModel* model = static_cast<CVMModel*>(net);
  *size = static_cast<unsigned long long>(model->GetSizeOfOutputType());
  API_END();
}

int CVMAPIGetInputTypeSize(void *net, unsigned long long *size) {
  API_BEGIN();
  CHECK_2_NOT_NULL(net, size);
  CVMModel* model = static_cast<CVMModel*>(net);
  *size = static_cast<unsigned long long>(model->GetSizeOfInputType());
  API_END();
}

int CVMAPIGetStorageSize(void *net, unsigned long long *gas) {
  API_BEGIN();
  CHECK_2_NOT_NULL(net, gas);
  CVMModel* model = static_cast<CVMModel*>(net);
  *gas = static_cast<unsigned long long>(model->GetStorageSize());
  API_END();
}

int CVMAPIGetGasFromModel(void *net, unsigned long long *gas) {
  API_BEGIN();
  CHECK_2_NOT_NULL(net, gas);
  CVMModel* model = static_cast<CVMModel*>(net);
  *gas = static_cast<unsigned long long>(model->GetOps());
  API_END();
}

int CVMAPIGetGasFromGraphFile(const char *graph_json, unsigned long long *gas) {
  API_BEGIN();
  string json_data(graph_json);
  auto f = cvm::runtime::Registry::Get("cvm.runtime.estimate_ops");
  int64_t ret = (*f)(json_data);
  *gas = static_cast<unsigned long long>(ret);
  API_END();
}

