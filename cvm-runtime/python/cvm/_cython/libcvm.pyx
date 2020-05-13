cimport ccvm
import numpy as np
from cvm._base import check_call
from cvm import CVMContext, kDLCPU, runtime_context


cdef class CVMRuntime:
    cdef void *network

    def __init__(self, bytes graph_json, bytes param_bytes,
            int device_type = kDLCPU,
            int device_id = 0):

        ctx = runtime_context(CVMContext(device_type, device_id))
        check_call(ccvm.CVMAPILoadModel(
            graph_json, len(graph_json),
            param_bytes, len(param_bytes),
            &self.network, ctx.device_type, ctx.device_id))

    def FreeModel(self):
        check_call(ccvm.CVMAPIFreeModel(self.network))

    def Inference(self, char *input_data):
        input_size = self.GetInputLength()
        output_size = self.GetOutputLength()
        output_data = bytes(output_size)
        output_type_size = self.GetOutputTypeSize()

        check_call(ccvm.CVMAPIInference(
            self.network, input_data, input_size, output_data))

        max_v = (1 << (output_type_size * 8 - 1))
        infer_result = []
        for i in range(0, output_size, output_type_size):
            int_val = int.from_bytes(output_data[i:i+output_type_size], byteorder='little')
            infer_result.append(int_val if int_val < max_v else \
                int_val - 2 * max_v)
        return infer_result

    def GetVersion(self, char *version):
        check_call(ccvm.CVMAPIGetVersion(self.network, version))

    def GetPreprocessMethod(self, char *method):
        check_call(ccvm.CVMAPIGetPreprocessMethod(self.network, method))

    def GetInputLength(self):
        cdef unsigned long long[1] csize
        check_call(ccvm.CVMAPIGetInputLength(self.network, csize))
        return csize[0]

    def GetOutputLength(self):
        cdef unsigned long long[1] csize
        check_call(ccvm.CVMAPIGetOutputLength(self.network, csize))
        return csize[0]

    def GetInputTypeSize(self):
        cdef unsigned long long[1] csize
        check_call(ccvm.CVMAPIGetInputTypeSize(self.network, csize))
        return csize[0]

    def GetOutputTypeSize(self):
        cdef unsigned long long[1] csize
        check_call(ccvm.CVMAPIGetOutputTypeSize(self.network, csize))
        return csize[0]

    def GetStorageSize(self):
        cdef unsigned long long[1] cgas
        check_call(ccvm.CVMAPIGetStorageSize(self.network, cgas))
        return cgas[0]

    def GetGasFromModel(self):
        cdef unsigned long long[1] cgas
        check_call(ccvm.CVMAPIGetGasFromModel(self.network, cgas))
        return cgas[0]

    def GetGasFromGraphFile(const char *graph_json):
        cdef unsigned long long[1] cgas
        check_call(ccvm.CVMAPIGetGasFromGraphFile(graph_json, cgas))
        return cgas[0]
