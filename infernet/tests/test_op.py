import mxnet as mx
from mxnet import ndarray as nd
n, m = 100, 1000
x = nd.random.randint(-127, 128, shape=(n, m))
print (nd.min(x), nd.max(x))
x = nd.array([[  1.,   2.,   3.,   4.], [  5.,   6.,   7.,   8.], [  9.,  10.,  11.,  12.]])
print (nd.slice(x, begin=(0,1), end=(2,4)))
#nd.slice()

import json
js = json.load(open('/home/tian/model_storage/yolo3_darknet53/data/symbol'))
for idx, x in enumerate(js['nodes']):
    if (x['name'] == 'strided_slice'):
        for input_idx in x['inputs']:
            child_id = input_idx[0]
            print (input_idx, child_id, js['attrs']['shape'][1][child_id])
        print (idx, x, js['attrs']['op_attrs'][1][idx], js['attrs']['shape'][1][idx])
