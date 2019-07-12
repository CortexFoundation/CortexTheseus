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

typedef unsigned long long* IntHandler;
typedef char* StringHandler;

int CVMAPILoadModel(const char *graph_json, int graph_strlen,
                          const char *param_bytes, int param_strlen,
                          void **net, // pass reference of network
                          int device_type, int device_id);
int CVMAPIFreeModel(void *net);
int CVMAPIInference(void *net,
                          char *input_data, int input_len,
                          StringHandler output_data);

int CVMAPIGetVersion(void *net, StringHandler version);
int CVMAPIGetPreprocessMethod(void *net, StringHandler method);

int CVMAPIGetInputLength(void *net, IntHandler size);
int CVMAPIGetOutputLength(void *net, IntHandler size);
int CVMAPIGetInputTypeSize(void *net, IntHandler size);
int CVMAPIGetOutputTypeSize(void *net, IntHandler size);

int CVMAPIGetStorageSize(void *net, IntHandler gas);
int CVMAPIGetGasFromModel(void *net, IntHandler gas);
int CVMAPIGetGasFromGraphFile(const char *graph_json, IntHandler gas);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // CVM_C_API_H_
