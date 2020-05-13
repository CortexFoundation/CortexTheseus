import ctypes
import os
import numpy as np

from .. import libinfo
from .._base import check_call
from .lib import _LIB
from .context import *

class CVMDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8),
                ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]
    CODE2STR = {
        0 : 'int',
        1 : 'uint',
        2 : 'float',
        4 : 'handle'
    }
    def __init__(self, type_str):
        super(CVMDataType, self).__init__()
        if isinstance(type_str, np.dtype):
            type_str = str(type_str)

        if not isinstance(type_str, str):
            raise RuntimeError("CVMDataType only support numpy.dtype " +
                "or string indicator like int32 instead of {}".format(
                    type(type_str)))

        if type_str == "bool":
            self.bits = 1
            self.code = 1
            self.lanes = 1
            return

        arr = type_str.split("x")
        head = arr[0]
        self.lanes = int(arr[1]) if len(arr) > 1 else 1
        bits = 32


        if head.startswith("int"):
            self.code = 0
            head = head[3:]
        elif head.startswith("uint"):
            self.code = 1
            head = head[4:]
        elif head.startswith("float"):
            self.code = 2
            head = head[5:]
        elif head.startswith("handle"):
            self.code = 4
            bits = 64
            head = ""
        elif head.startswith("custom"):
            low, high = head.find('['), head.find(']')
            if not low or not high or low >= high:
                raise ValueError("Badly formatted custom type string %s" % type_str)
            type_name = head[low + 1:high]
            self.code = _api_internal._datatype_get_type_code(type_name)
            head = head[high+1:]
        else:
            raise ValueError("Do not know how to handle type %s" % type_str)
        bits = int(head) if head else bits
        self.bits = bits

    def __repr__(self):
        if self.bits == 1 and self.lanes == 1:
            return "bool"
        if self.code in CVMDataType.CODE2STR:
            type_name = CVMDataType.CODE2STR[self.code]
        else:
            type_name = "custom[%s]" % \
                        _api_internal._datatype_get_type_name(self.code)
        x = "%s%d" % (type_name, self.bits)
        if self.lanes != 1:
            x += "x%d" % self.lanes
        return x

    def __eq__(self, other):
        return (self.bits == other.bits and
                self.code == other.code and
                self.lanes == other.lanes)

    def __ne__(self, other):
        return not self.__eq__(other)

class CVMArray(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p),
                ('ctx', CVMContext),
                ('ndim', ctypes.c_int),
                ("dtype", CVMDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]

class NDArrayBase(object):
    __slots__ = ["handle", "is_view"]
    def __init__(self, handle, is_view=False):
        self.handle = handle
        self.is_view = is_view

    def __del__(self):
        if not self.is_view and _LIB:
            check_call(_LIB.CVMArrayFree(self.handle))

    @property
    def _cvm_handle(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

CVMArrayHandle = ctypes.POINTER(CVMArray)

def c_array(ctype, values):
    return (ctype * len(values))(*values)

CVMStreamHandle = ctypes.c_void_p

class CVMByteArray(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_byte)),
                ("size", ctypes.c_size_t)]

    def tobytes(self):
        size = self.size
        res = bytearray(size)
        rptr = (ctypes.c_byte * size).from_buffer(res)
        if not ctypes.memmove(rptr, self.data, size):
            raise RuntimeError('memmove failed')
        return bytes(res)
