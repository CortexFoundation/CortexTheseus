import ctypes
from .common import context, cpu
from ._ctypes.ndarray import *
from ._ctypes.lib import _LIB

import numpy as np

class NDArray(NDArrayBase):
    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i] for i in range(self.handle.contents.ndim))

    @property
    def dtype(self):
        """Type of this array"""
        return str(self.handle.contents.dtype)

    @property
    def ctx(self):
        """context of this array"""
        return self.handle.contents.ctx

    @property
    def context(self):
        """context of this array"""
        return self.ctx

    def __hash__(self):
        return ctypes.cast(self.handle, ctypes.c_void_p).value

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Check object identity equality

        Parameters
        ----------
        other : object
            The other object to compare to

        Returns
        -------
        same : bool
            Whether other is same as self.
        """
        if not isinstance(other, NDArrayBase):
            return False
        return self.__hash__() == other.__hash__()

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArrayBase):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self.copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def copyfrom(self, source_array):
        """Peform an synchronize copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.

        Returns
        -------
        arr : NDArray
            Reference to self.
        """
        if isinstance(source_array, NDArrayBase):
            source_array.copyto(self)
            return self

        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=self.dtype)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(source_array)))
        t = CVMDataType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)

        if source_array.shape != shape:
            raise ValueError("array shape do not match the shape of NDArray {0} vs {1}".format(
                source_array.shape, shape))
        source_array = np.ascontiguousarray(source_array, dtype=dtype)
        assert source_array.flags['C_CONTIGUOUS']
        data = source_array.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(source_array.size * source_array.dtype.itemsize)
        check_call(_LIB.CVMArrayCopyFromBytes(self.handle, data, nbytes))
        return self

    def __repr__(self):
        res = "<cvm.NDArray shape={0} {1}>\n".format(self.shape, self.context)
        res += self.asnumpy().__repr__()
        return res

    def __str__(self):
        return str(self.asnumpy())

    def asnumpy(self):
        """Convert this array to numpy array

        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        t = CVMDataType(self.dtype)
        shape, dtype = self.shape, self.dtype
        if t.lanes > 1:
            shape = shape + (t.lanes,)
            t.lanes = 1
            dtype = str(t)
        np_arr = np.empty(shape, dtype=dtype)
        assert np_arr.flags['C_CONTIGUOUS']
        data = np_arr.ctypes.data_as(ctypes.c_void_p)
        nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
        check_call(_LIB.CVMArrayCopyToBytes(self.handle, data, nbytes))
        return np_arr

    def as_runtime_input(self):
        return self.asnumpy().tobytes()

    def copyto(self, target):
        if isinstance(target, CVMContext):
            target = empty(self.shape, self.dtype, target)
        if isinstance(target, NDArrayBase):
            check_call(_LIB.CVMArrayCopyFromTo(
                self.handle, target.handle, None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target


def numpyasarray(np_data):
    """Return a CVMArray representation of a numpy array.
    """
    data = np_data
    assert data.flags['C_CONTIGUOUS']
    arr = CVMArray()
    shape = c_array(ctypes.c_int64, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.strides = None
    arr.dtype = CVMDataType(np.dtype(data.dtype).name)
    arr.ndim = data.ndim
    # CPU device
    arr.ctx = context(1, 0)
    return arr, shape

def _make_array(handle):
    handle = ctypes.cast(handle, CVMArrayHandle)
    return NDArray(handle, False)

def empty(shape, dtype="int32", ctx=cpu()):
    shape = c_array(ctypes.c_int64, shape)
    ndim = ctypes.c_int(len(shape))
    handle = CVMArrayHandle()
    dtype = CVMDataType(dtype)
    check_call(_LIB.CVMArrayAlloc(
        shape, ndim,
        ctypes.c_int(dtype.code),
        ctypes.c_int(dtype.bits),
        ctypes.c_int(dtype.lanes),
        ctx.device_type,
        ctx.device_id,
        ctypes.byref(handle)))
    return _make_array(handle)

def array(arr, ctx=cpu()):
    """Create an array from source arr.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from

    ctx : CVMContext, optional
        The device context to create the array

    Returns
    -------
    ret : NDArray
        The created array
    """
    if not isinstance(arr, (np.ndarray, NDArray)):
        arr = np.array(arr, np.int32)
    return empty(arr.shape, arr.dtype, ctx).copyfrom(arr)

def save_param_dict(dict_data):
    data = []
    for k, v in dict_data.items():
        pk = ctypes.c_char_p(bytes(k, encoding='utf-8'))
        data.append(ctypes.cast(pk, ctypes.c_void_p))
        data.append(ctypes.cast(v.handle, ctypes.c_void_p))

    ret = CVMByteArray()
    arr = (ctypes.c_void_p * len(data))(*data)
    check_call(_LIB.CVMSaveParamsDict(ctypes.byref(arr), len(data), ctypes.byref(ret)))
    return ret.tobytes()

