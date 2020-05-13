import unittest
import mxnet as mx

import nnvm

from _base import *

class TestNull(TfmTest):
    op = mx.sym.var("data", shape=[1, 3, 2, 1])

    def test_validate(self):
        self._assert_equal(self.op, self.op, "validate")

    def test_rewrite(self):
        self._assert_equal(self.op, self.op, "rewrite")

    def test_fuse_transpose(self):
        self._assert_equal(self.op, self.op, "fuse_transpose")

    def test_compile(self):
        des = nnvm.sym.Variable('data', __shape__=[1, 3, 2, 1])
        self._assert_equal(self.op, des, "compile")

 
class TestTranspose(TfmTest):
    op = mx.sym.transpose(
            mx.sym.transpose(TestNull.op, axes=[0, 2, 3, 1]),
            axes=[3, 2, 1, 0])

    def test_validate(self):
        self._assert_equal(self.op, self.op, "validate")

        op1 = mx.sym.transpose(TestNull.op, axes=[])
        des = mx.sym.transpose(TestNull.op, axes=[3, 2, 1, 0])
        self._assert_equal(op1, des, "validate")

        op2 = mx.sym.transpose(TestNull.op)
        self._assert_equal(op2, des, "validate")

    def test_rewrite(self):
        self._assert_equal(self.op, self.op, "rewrite")

    def test_fuse_transpose(self):
        des = mx.sym.transpose(TestNull.op, axes=[1, 3, 2, 0])
        self._assert_equal(self.op, des, "fuse_transpose")


class TestElemwiseAdd(TfmTest):
    def test_fuse_transpose(self):
        x = mx.sym.var("x", shape=(1,2,3,4))
        op1 = mx.sym.transpose(x, axes=[1, 3, 2, 0])
        op2 = mx.sym.transpose(x, axes=[1, 3, 2, 0])
        op = mx.symbol.elemwise_add(op1, op2)

        des = mx.sym.transpose(
                mx.sym.elemwise_add(x, x),
                axes=[1, 3, 2, 0])

        self._assert_equal(op, des, "fuse_transpose")


class TestBroadcastAdd(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,3))
        y = mx.sym.var('y', shape=(2,1))
        ans = mx.sym.broadcast_add(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_add(x, y)
        self._assert_equal(ans, des, "compile")


class TestBroadcastDiv(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,3))
        y = mx.sym.var('y', shape=(2,1))
        ans = mx.sym.broadcast_div(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_div(x, y)
        self._assert_equal(ans, des, "compile")


class TestBroadcastMul(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,3))
        y = mx.sym.var('y', shape=(2,1))
        ans = mx.sym.broadcast_mul(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_mul(x, y)
        self._assert_equal(ans, des, "compile")


class TestBroadcastGreater(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,3))
        y = mx.sym.var('y', shape=(2,1))
        ans = mx.sym.broadcast_greater(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_greater(x, y)
        self._assert_equal(ans, des, "compile")


class TestAbs(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,4))
        ans = mx.sym.abs(x)

        x = nnvm.sym.Variable('x', __shape__=(2,4))
        des = nnvm.sym.abs(x)
        self._assert_equal(ans, des, 'compile')


class TestMinimum(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 4))
        y = mx.sym.var('y', shape=(2, 4))
        ans = mx.sym.minimum(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 4))
        y = nnvm.sym.Variable('y', __shape__=(2, 4))
        des = nnvm.sym.broadcast_min(x, y)
        self._assert_equal(ans, des, 'compile')


class TestArgmax(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 4))
        ans = mx.sym.argmax(x)

        x = nnvm.sym.Variable('x', __shape__=(2, 4))
        des = nnvm.sym.argmax(x,axis=0,keepdims=False)

        self._assert_equal(ans, des, 'compile')


class TestArgmin(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 4))
        ans = mx.sym.argmin(x)

        x = nnvm.sym.Variable('x', __shape__=(2, 4))
        des = nnvm.sym.argmin(x,axis=0,keepdims=False)

        self._assert_equal(ans, des, 'compile')


class TestActivation(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 4))
        ans= mx.sym.Activation(data=x, act_type='relu')

        x = nnvm.sym.Variable('x', __shape__=(2, 4))
        des= nnvm.sym.relu(data=x)
        self._assert_equal(ans, des, 'compile')


class TestMaximum(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 4))
        y = mx.sym.var('y', shape=(2, 4))
        ans = mx.sym.maximum(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 4))
        y = nnvm.sym.Variable('y', __shape__=(2, 4))
        des = nnvm.sym.broadcast_max(x, y)
        self._assert_equal(ans, des, 'compile')


class TestCustom(TfmTest):
    def test_custom(self):
        x = mx.sym.var('x', shape=(2,4))
        ans = mx.sym.Custom(x, precision=8, op_type='cvm_clip',name='customout')

        x = nnvm.sym.Variable('x', __shape__=(2,4))
        des = nnvm.sym.cvm_clip(x, precision=8)
        self._assert_equal(ans, des, 'compile')


class TestClip(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 3))
        ans = mx.sym.clip(x, 1, 10)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        des = nnvm.sym.clip(x, a_min=1, a_max=10)
        self._assert_equal(ans, des, 'compile')


class TestExpandDims(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 3, 4))
        ans = mx.sym.expand_dims(x, axis=1)

        x = nnvm.sym.Variable('x', __shape__=(2, 3, 4))
        des = nnvm.sym.expand_dims(x, axis=1)
        self._assert_equal(ans, des, 'compile')


class TestEmbedding(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 2))
        y = mx.sym.var('y', shape=(4, 5))
        ans = mx.sym.Embedding(x, y, 4, 5)

        x = nnvm.sym.Variable('x', __shape__=(2, 2))
        y = nnvm.sym.Variable('y', __shape__=(4, 5))
        des = nnvm.sym.take(y, x, axis=0)
        self._assert_equal(ans, des, 'compile')


class TestBroadcastSub(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 3))
        y = mx.sym.var('y', shape=(2, 1))
        ans = mx.sym.broadcast_sub(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_sub(x, y)
        self._assert_equal(ans, des, "compile")


class TestRepeat(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 2))
        ans = mx.sym.repeat(x, repeats=2)

        x = nnvm.sym.Variable('x', __shape__=(2, 2))
        des = nnvm.sym.repeat(x, repeats=2, axis=0)
        self._assert_equal(ans, des, 'compile')


class TestSliceLike(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(3, 4))
        y = mx.sym.var('y', shape=(2, 3))
        ans = mx.sym.slice_like(x, y, axes=(0, 0))

        x = nnvm.sym.Variable('x', __shape__=(3, 4))
        y = nnvm.sym.Variable('y', __shape__=(2, 3))
        des = nnvm.sym.slice_like(x, y, axis=(0, 0))
        self._assert_equal(ans, des, 'compile')


class TestBoxNms(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(4, 6))
        ans = mx.sym.contrib.box_nms(x)

        x = nnvm.sym.Variable('x', __shape__=(4, 6))
        des = nnvm.sym.get_valid_counts(x, score_threshold=0)
        des = nnvm.sym.non_max_suppression(des[1], des[0],
                iou_threshold=0.5,
                force_suppress=False, top_k=-1,
                coord_start=2,
                score_index=1, id_index=-1,
                return_indices=False, invalid_to_bottom=True)
        self._assert_equal(ans, des, ['prepare_for_compile', 'compile'])



class TestBroadcastTo(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(1,3))
        ans = mx.sym.broadcast_to(x, shape=(2,3))

        x = nnvm.sym.Variable('x', __shape__=(1,3))
        des = nnvm.sym.broadcast_to(x, shape=(2,3))
        self._assert_equal(ans, des, 'compile')


class TestUnSampling(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(1, 1, 3, 3))
        ans = mx.sym.UpSampling(x, scale=2, sample_type='nearest')

        x = nnvm.sym.Variable('x', __shape__=(1, 1, 3, 3))
        des = nnvm.sym.upsampling(x, scale=2)
        self._assert_equal(ans, des, 'compile')


class TestConcat(TfmTest):
    def test_fuse_transpose(self):
        x = mx.sym.var('x', shape=(1, 3, 244, 244))
        op1 = mx.sym.transpose(x, axes=[3, 2, 1, 0])
        op2 = mx.sym.transpose(x, axes=[])
        op = mx.sym.Concat(op1, op2, dim=1)

        des = mx.sym.transpose(mx.sym.concat(x, x, dim=2), axes=[3,2,1,0])

        self._assert_equal(op, des,
                ["validate", "fuse_transpose"])

    def test_compile(self):
        x = mx.sym.var('x', shape=(2, 2))
        y = mx.sym.var('y', shape=(3, 2))
        z = mx.sym.var('z', shape=(3, 2))
        ans = mx.sym.Concat(x, y, z, dim=0)

        x = nnvm.sym.Variable('x', __shape__=(2, 2))
        y = nnvm.sym.Variable('y', __shape__=(3, 2))
        z = nnvm.sym.Variable('z', __shape__=(3, 2))
        des = nnvm.sym.concatenate(x, y, z, axis=0)
        self._assert_equal(ans, des, 'compile')


class TestSlice(TfmTest):
    def test_compile(self):
        data = mx.sym.var('data', shape=(20,))
        op = mx.sym.slice(data, begin=(0,), end=(6,))

        datan = nnvm.sym.Variable('data', __shape__=(20,))
        des = nnvm.sym.strided_slice(datan, begin=(0,), end=(6,))

        self._assert_equal(op, des, 'compile')


class TestReshape(TfmTest):
    def test_compile(self):
        data = mx.sym.var('data', shape=(24,))
        op = mx.sym.reshape(data, shape=(1, 2, 3, 4))

        datan = nnvm.sym.Variable('data', __shape__=(24,))
        des = nnvm.sym.reshape(datan, shape=(1, 2, 3, 4))

        self._assert_equal(op, des, 'compile')


class TestFlatten(TfmTest):
    def test_compile(self):
        data = mx.sym.var('data', shape=(3, 3))
        ans = mx.sym.Flatten(data)

        data = nnvm.sym.Variable('data', __shape__=(3, 3))
        des = nnvm.sym.flatten(data)
        self._assert_equal(ans, des, 'compile')


class TestFullyConnected(TfmTest):
    def test_compile(self):
        data = mx.sym.var('data', shape=(4, 3, 3, 3))
        weight = mx.sym.var('weight', shape=(4, 27))
        bias = mx.sym.var('bias', shape=(4,))
        ans = mx.sym.FullyConnected(data, weight, bias, 4, flatten=True, no_bias=False)

        a = nnvm.sym.Variable('data', __shape__=(4, 3, 3, 3))
        b = nnvm.sym.Variable('weight', __shape__=(4, 27))
        c = nnvm.sym.Variable('bias', __shape__=(4,))
        d = nnvm.sym.dense(nnvm.sym.flatten(a), b, c, units=4, use_bias=True)
        self._assert_equal(ans, d, 'compile')


class TestPooling(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(4, 2, 3, 5))
        ans = mx.sym.Pooling(x, kernel=(2, 2))

        x = nnvm.sym.Variable('x', __shape__=(4, 2, 3, 5))
        des = nnvm.sym.max_pool2d(x, pool_size=(2, 2), strides=(1, 1),
                padding=(0, 0), ceil_mode=False)
        self._assert_equal(ans, des, 'compile')


class TestConvolution(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(3, 2, 2, 2))
        y = mx.sym.var('y', shape=(1, 2, 2, 2))
        z = mx.sym.var('z', shape=(1,))
        ans = mx.sym.Convolution(x, y, z, kernel=(2, 2), num_filter=1)

        x = nnvm.sym.Variable('x', __shape__=(3, 2, 2, 2))
        y = nnvm.sym.Variable('y', __shape__=(1, 2, 2, 2))
        z = nnvm.sym.Variable('z', __shape__=(1, ))
        des = nnvm.sym.conv2d(x, y, z, kernel_size=(2, 2), channels=1,
                strides=(1, 1), padding=(0, 0), layout='NCHW',
                kernel_layout='OIHW', use_bias=True, groups=1,
                dilation=(1, 1))
        self._assert_equal(ans, des, 'compile')

if __name__ == "__main__":
    import sys
    unittest.main(argv=sys.argv, verbosity=5)
