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

#ifdef __cplusplus
extern "C" {
#endif

const int SUCCEED = 0;
const int ERROR_LOGIC = 1;
const int ERROR_RUNTIME = 2;

int CVMAPILoadModel(const char *graph_json, int graph_strlen,
                          const char *param_bytes, int param_strlen,
                          void **net, // pass reference of network
                          int device_type, int device_id);
int CVMAPIFreeModel(void *net);
int CVMAPIInference(void *net,
                          char *input_data, int input_len,
                          char *output_data);

int CVMAPIGetVersion(void *net, char *version);
int CVMAPIGetPreprocessMethod(void *net, char *method);

int CVMAPIGetInputLength(void *net, unsigned long long *size);
int CVMAPIGetOutputLength(void *net, unsigned long long *size);
int CVMAPIGetInputTypeSize(void *net, unsigned long long *size);
int CVMAPIGetOutputTypeSize(void *net, unsigned long long *size);

int CVMAPIGetStorageSize(void *net, unsigned long long *gas);
int CVMAPIGetGasFromModel(void *net, unsigned long long *gas);
int CVMAPIGetGasFromGraphFile(const char *graph_json, unsigned long long *gas);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // CVM_C_API_H_
