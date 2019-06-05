import mxnet as mx
from mxnet import ndarray as nd
import numpy as np

a = nd.arange(0, stop=27).reshape((3,3,3))
#c = nd.transpose(a, axes=(0, 2, 3, 1))
c = nd.max(a, axis=(2))
print(c.shape)
#print(a)
print(c.asnumpy().flatten())
#nd.slice()
np.save("./tests/out.npy", c.asnumpy().astype("int32"))

import json
js = json.load(open('/home/tian/model_storage/sentiment_trec/data/symbol'))
for idx, x in enumerate(js['nodes']):
    if (x['name'].startswith('take')):
        for input_idx in x['inputs']:
            child_id = input_idx[0]
            print (input_idx, child_id, js['attrs']['shape'][1][child_id])
        print (idx, x, js['attrs']['op_attrs'][1][idx].replace('"', '\\"'), js['attrs']['shape'][1][idx])
