import numpy as np
from tensorflow.python.framework import dtypes

def tensor_shape_proto_to_list(shape):
    shp = [d.size if d.size > 0 else 1 for d in shape.dim]
    return shp
    #  return (1,) if len(shp) == 0 else shp

def tensor_type_to_numpy(dtype):
    return dtypes.as_dtype(dtype).as_numpy_dtype

def tensor_to_numpy(tensor):
    shape = tensor_shape_proto_to_list(tensor.tensor_shape)
    num_elements = np.prod(shape, dtype=np.int64)
    tensor_dtype = dtype = tensor_type_to_numpy(tensor.dtype)

    if tensor.tensor_content:
        return (np.frombuffer(tensor.tensor_content, dtype=dtype).copy()
                .reshape(shape))
    elif tensor_dtype == dtypes.float16 or tensor_dtype == dtypes.bfloat16:
        # the half_val field of the TensorProto stores the binary representation
        # of the fp16: we need to reinterpret this as a proper float16
        if len(tensor.half_val) == 1:
            tmp = np.array(tensor.half_val[0], dtype=np.uint16)
            tmp.dtype = tensor_dtype.as_numpy_dtype
            return np.repeat(tmp, num_elements).reshape(shape)
        else:
            tmp = np.fromiter(tensor.half_val, dtype=np.uint16)
            tmp.dtype = tensor_dtype.as_numpy_dtype
            return tmp.reshape(shape)
    elif tensor_dtype == dtypes.float32:
        if len(tensor.float_val) == 1:
            return np.repeat(
                    np.array(tensor.float_val[0], dtype=dtype),
                    num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.float_val, dtype=dtype).reshape(shape)
    elif tensor_dtype == dtypes.float64:
        if len(tensor.double_val) == 1:
            return np.repeat(
                    np.array(tensor.double_val[0], dtype=dtype),
                    num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.double_val, dtype=dtype).reshape(shape)
    elif tensor_dtype in [
        dtypes.int32, dtypes.uint8, dtypes.uint16, dtypes.int16, dtypes.int8,
        dtypes.qint32, dtypes.quint8, dtypes.qint8, dtypes.qint16, dtypes.quint16
    ]:
        if len(tensor.int_val) == 1:
            return np.repeat(np.array(tensor.int_val[0], dtype=dtype),
                    num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.int_val, dtype=dtype).reshape(shape)
    elif tensor_dtype == dtypes.int64:
        if len(tensor.int64_val) == 1:
            return np.repeat(
                    np.array(tensor.int64_val[0], dtype=dtype),
                    num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.int64_val, dtype=dtype).reshape(shape)
    elif tensor_dtype == dtypes.string:
        if len(tensor.string_val) == 1:
            return np.repeat(
                    np.array(tensor.string_val[0], dtype=dtype),
                    num_elements).reshape(shape)
        else:
            return np.array(
                    [x for x in tensor.string_val], dtype=dtype).reshape(shape)
    elif tensor_dtype == dtypes.complex64:
        it = iter(tensor.scomplex_val)
        if len(tensor.scomplex_val) == 2:
            return np.repeat(
                    np.array(
                        complex(tensor.scomplex_val[0], tensor.scomplex_val[1]),
                        dtype=dtype), num_elements).reshape(shape)
        else:
            return np.array(
                    [complex(x[0], x[1]) for x in zip(it, it)],
                    dtype=dtype).reshape(shape)
    elif tensor_dtype == dtypes.complex128:
        it = iter(tensor.dcomplex_val)
        if len(tensor.dcomplex_val) == 2:
            return np.repeat(
                    np.array(
                        complex(tensor.dcomplex_val[0], tensor.dcomplex_val[1]),
                        dtype=dtype), num_elements).reshape(shape)
        else:
            return np.array(
                    [complex(x[0], x[1]) for x in zip(it, it)],
                    dtype=dtype).reshape(shape)
    elif tensor_dtype == dtypes.bool:
        if len(tensor.bool_val) == 1:
            return np.repeat(np.array(tensor.bool_val[0], dtype=dtype),
                num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.bool_val, dtype=dtype).reshape(shape)
    else:
        raise TypeError("Unsupported tensor type: %s" % tensor.dtype)
