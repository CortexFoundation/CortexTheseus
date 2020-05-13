import ctypes
import os
import numpy as np

from .. import libinfo
from .._base import check_call
from ..common import cpu, runtime_context
from .lib import _LIB

NetworkHandle = ctypes.c_void_p

def CVMAPILoadModel(json_str, param_bytes, ctx=None):
    if ctx is None:
        ctx = cpu(0)
    ctx = runtime_context(ctx)

    net = NetworkHandle()
    if isinstance(json_str, str):
        json_str = json_str.encode("utf-8")
    check_call(_LIB.CVMAPILoadModel(
        ctypes.c_char_p(json_str), ctypes.c_int(len(json_str)),
        ctypes.c_char_p(param_bytes), ctypes.c_int(len(param_bytes)),
        ctypes.byref(net),
        ctx.device_type, ctx.device_id))
    return net

def CVMAPIFreeModel(net):
    check_call(_LIB.CVMAPIFreeModel(net))

def CVMAPIGetInputLength(net):
    size = ctypes.c_ulonglong()
    check_call(_LIB.CVMAPIGetInputLength(net, ctypes.byref(size)))
    return size.value

def CVMAPIGetInputTypeSize(net):
    size = ctypes.c_ulonglong()
    check_call(_LIB.CVMAPIGetInputTypeSize(net, ctypes.byref(size)))
    return size.value

def CVMAPIGetOutputLength(net):
    size = ctypes.c_ulonglong()
    check_call(_LIB.CVMAPIGetOutputLength(net, ctypes.byref(size)))
    return size.value

def CVMAPIGetOutputTypeSize(net):
    size = ctypes.c_ulonglong()
    check_call(_LIB.CVMAPIGetOutputTypeSize(net, ctypes.byref(size)))
    return size.value

def CVMAPIInference(net, input_data):
    osize = CVMAPIGetOutputLength(net)

    output_data = bytes(osize)
    check_call(_LIB.CVMAPIInference(
        net,
        ctypes.c_char_p(input_data), ctypes.c_int(len(input_data)),
        ctypes.c_char_p(output_data)))

    otype_size = CVMAPIGetOutputTypeSize(net)
    ret = []
    max_v = (1 << (otype_size * 8  - 1))
    for i in range(0, osize, otype_size):
        int_val = int.from_bytes(
            output_data[i:i+otype_size], byteorder='little')
        ret.append(int_val if int_val < max_v else int_val - 2 * max_v)

    return ret
