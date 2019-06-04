import mxnet as mx
from mxnet import ndarray as nd
import numpy as np

a = nd.arange(0, stop=60).reshape((5, 2, 2, 3))
#c = nd.transpose(a, axes=(0, 2, 3, 1))
c = nd.expand_dims(a, axis=(-1))
print(c.shape)
print(c.asnumpy().flatten())
#nd.slice()

import json
js = json.load(open('/home/tian/model_storage/sentiment_trec/data/symbol'))
for idx, x in enumerate(js['nodes']):
    if (x['name'].startswith('take')):
        for input_idx in x['inputs']:
            child_id = input_idx[0]
            print (input_idx, child_id, js['attrs']['shape'][1][child_id])
        print (idx, x, js['attrs']['op_attrs'][1][idx].replace('"', '\\"'), js['attrs']['shape'][1][idx])
