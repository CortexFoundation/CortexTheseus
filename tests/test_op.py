import mxnet as mx
from mxnet import ndarray as nd
n, m = 100, 1000
x = nd.random.randint(-127, 128, shape=(n, m))
print (nd.min(x), nd.max(x))
x = nd.array([[  1.,   2.,   3.,   4.], [  5.,   6.,   7.,   8.], [  9.,  10.,  11.,  12.]])
print (nd.slice(x, begin=(0,1), end=(2,4)))
#nd.slice()
