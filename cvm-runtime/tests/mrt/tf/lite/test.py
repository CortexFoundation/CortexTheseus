import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.keras import preprocessing
import numpy as np
import os

# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')


cat_img = os.path.expanduser(
    "~/.keras/datasets/cats_and_dogs_filtered/validation/cats/cat.2000.jpg")
image = tf.keras.preprocessing.image.load_img(cat_img, target_size=(224, 224))
print (image)
data = tf.keras.preprocessing.image.img_to_array(image) / 255.
data = np.expand_dims(data, axis=0)
print (data.shape)

label_path = "/data/tfmodels/lite/DenseNet/labels.txt"
with open(label_path, "r") as f:
    lines = f.readlines()
    labels = {i:l for i, l in enumerate(lines)}


with tf.Session() as sess:
    graph_def = graph_pb2.GraphDef()
    model_path = "/data/tfmodels/mobilenet/model.pb"
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    # sess.graph.as_default()
    tf.import_graph_def(graph_def)

    #  name = "import/conv1_1/Conv2D:0"
    #  name = "import/conv1_bn_1/FusedBatchNorm:0"
    #  name = "import/conv_dw_1_1/depthwise:0"
    #  name = "import/conv_pw_13_relu_1/Relu6:0"
    #  name = "import/global_average_pooling2d_1/Mean:0"
    #  name = "import/reshape_1_1/Reshape:0"
    name = "import/act_softmax_1/Softmax:0"

    output_tensor = sess.graph.get_tensor_by_name(name)
    # input name : import/input_1_1
    print (output_tensor.name)
    res = sess.run(output_tensor, {'import/input_1_1:0': data})
    print (res.shape)
    print (res.reshape((-1,))[:10])
    argmax = res.flatten().argmax()
    print (argmax, labels[argmax] if argmax < 1000 else None)
    print (res.flatten()[argmax])
    #  np.save("/tmp/tf.batchnorm.npy", res)
