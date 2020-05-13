import cvm
from cvm import nd
import numpy as np
import time

npa = np.array([1,2,3,4])
nda = nd.array(npa, ctx=cvm.gpu())
print(nda.asnumpy(), nda)
npb = np.array([5,6,7,8])
ndb = nd.array(npb)

data = {'a':nda, 'b':ndb}
print(data.items())

ret = nd.save_param_dict(data)


import mxnet as mx
print (mx.nd.array(np.zeros((1,2,3)), mx.gpu()))
print (mx.nd.zeros((1,2,3), mx.gpu()))
