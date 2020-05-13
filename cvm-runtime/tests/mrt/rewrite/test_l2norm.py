import numpy as np

import mxnet as mx
from mxnet import ndarray as nd

def get_norm(data):
    data = data.asnumpy()
    ndims = np.product(data.shape)
    data = np.reshape(data, (ndims,))
    norm = np.linalg.norm(data)
    return norm, data

def verify_l2normalization_rewrite(shape, eps, mode):
    assert len(shape) == 4 # NCHW
    data_np = np.random.uniform(size=shape)
    x = nd.array(data_np)

    # org op
    y = nd.L2Normalization(x, eps=eps, mode=mode)

    # rewrite op
    z = nd.broadcast_mul(x, x)
    if mode == "channel":
        axis = [1]
    elif mode == "instance":
        axis = [1,2,3]
    elif mode == "spatial":
        axis = [2,3]
    else:
        assert "not valid `mode` type: %s" % mode
    z = nd.sum(z, axis=axis)
    eps_tensor = nd.array([eps])
    z = nd.broadcast_add(z, eps_tensor)
    z = nd.sqrt(z)
    for i in axis:
        z = nd.expand_dims(z, axis=i)
        z = nd.repeat(z, repeats=shape[i], axis=i)
    z = nd.broadcast_div(x, z)

    # compare
    assert z.shape == y.shape
    zn, zp = get_norm(z)
    yn, yp = get_norm(y)
    rn = np.linalg.norm(zp-yp)
    print(zn, yn, rn)

def test_l2normalization_rewrite():
    verify_l2normalization_rewrite((16, 512, 64, 64), 0, "channel")
    verify_l2normalization_rewrite((1, 512, 64, 64), 1e-5, "channel")
    verify_l2normalization_rewrite((16, 512, 64, 64), 1e-5, "channel")
    verify_l2normalization_rewrite((16, 512, 64, 64), 1, "channel")

    verify_l2normalization_rewrite((16, 512, 64, 64), 0, "instance")
    verify_l2normalization_rewrite((1, 512, 64, 64), 1e-5, "instance")
    verify_l2normalization_rewrite((16, 512, 64, 64), 1e-5, "instance")
    verify_l2normalization_rewrite((16, 512, 64, 64), 1, "instance")

    verify_l2normalization_rewrite((16, 512, 64, 64), 0, "spatial")
    verify_l2normalization_rewrite((1, 512, 64, 64), 1e-5, "spatial")
    verify_l2normalization_rewrite((16, 512, 64, 64), 1e-5, "spatial")
    verify_l2normalization_rewrite((16, 512, 64, 64), 1, "spatial")

if __name__ == '__main__':
    test_l2normalization_rewrite()
