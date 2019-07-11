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
  ERROR_LOGIC,
  ERROR_RUNTIME,
  ERROR_UNKNOWN
} ;

typedef void* ModelHandler;
typedef unsigned long long* IntHandler;
typedef char* StringHandler;

enum CVMStatus CVMAPILoadModel(const char *graph_json, int graph_strlen,
                          const char *param_bytes, int param_strlen,
                          ModelHandler *net, // pass reference of network
                          int device_type, int device_id);
enum CVMStatus CVMAPIFreeModel(ModelHandler net);
enum CVMStatus CVMAPIInference(ModelHandler net, char *input_data, StringHandler output_data);

enum CVMStatus CVMAPIGetVersion(ModelHandler net, StringHandler version);
enum CVMStatus CVMAPIGetPreprocessMethod(ModelHandler net, StringHandler method);

enum CVMStatus CVMAPIGetInputLength(ModelHandler net, IntHandler size);
enum CVMStatus CVMAPIGetOutputLength(ModelHandler net, IntHandler size);
enum CVMStatus CVMAPIGetInputTypeSize(ModelHandler net, IntHandler size);
enum CVMStatus CVMAPIGetOutputTypeSize(ModelHandler net, IntHandler size);

enum CVMStatus CVMAPIGetStorageSize(ModelHandler net, IntHandler gas);
enum CVMStatus CVMAPIGetGasFromModel(ModelHandler net, IntHandler gas);
enum CVMStatus CVMAPIGetGasFromGraphFile(const char *graph_json, IntHandler gas);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // CVM_C_API_H_
