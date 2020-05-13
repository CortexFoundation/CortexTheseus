
cdef extern from "cvm/c_api.h":
    int CVMAPILoadModel(const char *graph_json, int graph_strlen,
        const char *param_bytes, int param_strlen,
        void **net,
        int device_type, int device_id)
    int CVMAPIFreeModel(void *net)
    int CVMAPIInference(void *net,
        char *input_data, int input_len,
        char *output_data)

    int CVMAPIGetVersion(void *net, char *version)
    int CVMAPIGetPreprocessMethod(void *net, char *method)

    int CVMAPIGetInputLength(void *net, unsigned long long *size)
    int CVMAPIGetOutputLength(void *net, unsigned long long *size)
    int CVMAPIGetInputTypeSize(void *net, unsigned long long *size)
    int CVMAPIGetOutputTypeSize(void *net, unsigned long long *size)

    int CVMAPIGetStorageSize(void *net, unsigned long long *gas)
    int CVMAPIGetGasFromModel(void *net, unsigned long long *gas)
    int CVMAPIGetGasFromGraphFile(const char *graph_json, unsigned long long *gas)
