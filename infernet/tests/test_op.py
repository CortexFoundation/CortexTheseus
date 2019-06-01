import mxnet as mx
from mxnet import ndarray as nd
import numpy as np

n, m = 100, 1000
ishape = (16, 169, 3, 25)
a = nd.arange(0, stop=60).reshape((5, 2, 2, 3))
b = nd.zeros(shape=(4, 1, 1, 2))
c = nd.slice_like(a, b, axes=(0, 2, 3))
c.shape
print(c.asnumpy().flatten())
#nd.slice()

import json
js = json.load(open('/home/kaihuo/model_storage/yolo3_darknet53/data/symbol'))
for idx, x in enumerate(js['nodes']):
    if (x['name'] == 'strided_slice'):
        for input_idx in x['inputs']:
            child_id = input_idx[0]
            print (input_idx, child_id, js['attrs']['shape'][1][child_id])
        print (idx, x, js['attrs']['op_attrs'][1][idx], js['attrs']['shape'][1][idx])
