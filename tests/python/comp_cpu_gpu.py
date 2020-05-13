from cvm.base import CVMContext, CPU, GPU, FORMAL
from cvm.dll import load_model, load_np_data
from cvm.dll import CVMAPILoadModel, CVMAPIInference, CVMAPIGetOutputLength
from cvm import utils

import os

# CVMContext.set_global(CPU)
CVMContext.set_global(GPU)
# CVMContext.set_global(FORMAL)

# model_root = "/home/serving/tvm-cvm/data/jetson/"
# model_root = "/tmp/ssd_512_mobilenet1.0_coco_tfm/"
# model_root = "/data/std_out/resnet50_v2"

for model in os.listdir('/data/std_out/'):
    if model[0:6] == 'random':
        continue
    model_root = os.path.join('/data/std_out/', model)
    print("model_root === ", model_root)
    #model_root = "/data/std_out/ssd"
    try:
        json, params = load_model(os.path.join(model_root, "symbol"),
                         os.path.join(model_root, "params"))
    except:
        print("failed load model {}".format(model_root))
        continue

    net = CVMAPILoadModel(json, params)
    print(CVMAPIGetOutputLength(net))

    data_path = os.path.join(model_root, "data.npy")
    data = load_np_data(data_path)

    out = CVMAPIInference(net, data)

    # utils.classification_output(out)
    utils.detection_output(out)

