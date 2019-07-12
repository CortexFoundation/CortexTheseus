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
  ERROR_LOGIC = 1,
  ERROR_RUNTIME = 2,
  ERROR_UNKNOWN = 3
} ;

typedef void* ModelHandler;
typedef unsigned long long* IntHandler;
typedef char* StringHandler;

enum CVMStatus CVMAPILoadModel(const char *graph_json, int graph_strlen,
                          const char *param_bytes, int param_strlen,
                          void **net, // pass reference of network
                          int device_type, int device_id);
enum CVMStatus CVMAPIFreeModel(void *net);
enum CVMStatus CVMAPIInference(void *net,
                          char *input_data, int input_len,
                          StringHandler output_data);

enum CVMStatus CVMAPIGetVersion(void *net, StringHandler version);
enum CVMStatus CVMAPIGetPreprocessMethod(void *net, StringHandler method);

enum CVMStatus CVMAPIGetInputLength(void *net, IntHandler size);
enum CVMStatus CVMAPIGetOutputLength(void *net, IntHandler size);
enum CVMStatus CVMAPIGetInputTypeSize(void *net, IntHandler size);
enum CVMStatus CVMAPIGetOutputTypeSize(void *net, IntHandler size);

enum CVMStatus CVMAPIGetStorageSize(void *net, IntHandler gas);
enum CVMStatus CVMAPIGetGasFromModel(void *net, IntHandler gas);
enum CVMStatus CVMAPIGetGasFromGraphFile(const char *graph_json, IntHandler gas);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // CVM_C_API_H_
