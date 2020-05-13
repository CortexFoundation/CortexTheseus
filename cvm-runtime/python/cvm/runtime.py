from ._ctypes.runtime import CVMAPILoadModel, CVMAPIFreeModel
from ._ctypes.runtime import CVMAPIGetInputLength, CVMAPIGetInputTypeSize
from ._ctypes.runtime import CVMAPIInference
from ._ctypes.runtime import CVMAPIGetOutputLength, CVMAPIGetOutputTypeSize

try:
    from ._cy3 import libcvm
except ImportError:
    pass
