/*!
 *  Copyright (c) 2016 by Contributors
 * \file cvm/c_api.h
 * \brief C API of CVM symbolic construction and pass.
 *  Enables construction and transformation of Graph
 *  in any other host languages.
 */
#ifndef CVM_C_API_H_
#define CVM_C_API_H_

/*! \brief CVM_DLL prefix for windows */
#ifdef _WIN32
#ifdef CVM_EXPORTS
#define CVM_DLL __declspec(dllexport)
#else
#define CVM_DLL __declspec(dllimport)
#endif
#else
#define CVM_DLL __attribute__((visibility("default")))
#endif

/*
 * CVMAPIInterface
 * parameters : input, output or attributes
 * returns    : API execute status
 *            0 stands for succeed;
 *            -1 stands for logic error;
 *            -2 stands for runtime error;
 */
enum CVMStatus {
  SUCCEED = 0,
  ERROR_LOGIC,
  ERROR_RUNTIME,
  ERROR_UNKNOWN
};

typedef void* ModelHandler;
typedef long long* IntHandler;
typedef char* StringHandler;

#ifdef __cplusplus
extern "C" {
#endif

CVMStatus CVMAPILoadModel(const char *graph_json, int graph_strlen,
                          const char *param_bytes, int param_strlen,
                          ModelHandler *net, // pass reference of network
                          int device_type, int device_id);
CVMStatus CVMAPIFreeModel(ModelHandler net);
CVMStatus CVMAPIInference(ModelHandler net, char *input_data, StringHandler output_data);

CVMStatus CVMAPIGetVersion(ModelHandler net, StringHandler version);
CVMStatus CVMAPIGetPreprocessMethod(ModelHandler net, StringHandler method);

CVMStatus CVMAPIGetInputLength(ModelHandler net, IntHandler size);
CVMStatus CVMAPIGetOutputLength(ModelHandler net, IntHandler size);
CVMStatus CVMAPIGetInputTypeSize(ModelHandler net, IntHandler size);
CVMStatus CVMAPIGetOutputTypeSize(ModelHandler net, IntHandler size);

CVMStatus CVMAPIGetStorageSize(ModelHandler net, IntHandler gas);
CVMStatus CVMAPIGetGasFromModel(ModelHandler net, IntHandler gas);
CVMStatus CVMAPIGetGasFromGraphFile(const char *graph_json, IntHandler gas);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // CVM_C_API_H_
