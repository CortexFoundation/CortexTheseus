import os
import ctypes

from ..libinfo import find_lib_path

_LIB_PATH = find_lib_path()
_LIB_NAME = os.path.basename(_LIB_PATH[0])
_LIB = ctypes.CDLL(_LIB_PATH[0], ctypes.RTLD_GLOBAL)
