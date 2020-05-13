from cvm.base import CVMContext, CPU, GPU, FORMAL

from cvm.utils import load_model, load_np_data
from cvm.dll import CVMAPILoadModel, CVMAPIInference, CVMAPIGetOutputLength
from cvm import utils
from cvm import video
import json

import os
import sys
import numpy as np
import cv2
import time

from gluoncv import data as gdata

CVMContext.set_global(GPU)

data_mode = "videos"
source    = "/data/wlt/videos/video.MOV"
dest      = None
dest_type = video.VideoType.MP4

threshold = 0.5
text_font = 2
preview   = False

def mnist_transform(x):
    x = x.astype('float32').mean(2)
    x = (255 - x) * 127 / 255
    x = x.clip(-127, 127).astype(np.int8).reshape((1, 1, 28, 28))
    x = x.tobytes()
    # x = bytes([0 if d < 64 else 127 for d in x])
    x = bytes([10 * round(d // 10) for d in x])
    return x

def mnist_out(x):
    number = np.argmax(x)
    return [("Inference result: %d" % (number), (0, 0), (1, 1))]

def ssd_transform(x,
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]):
    x = x.astype('float32') / 255
    x = (x - np.array(mean)) / np.array(std)
    x = np.transpose(x, (2, 0, 1)) * 48.1060606060606
    x = x.clip(-127, 127).astype(np.int8).reshape((1, 3, 512, 512))
    return x.tobytes()

def ssd_out(x):
    x = x.reshape(-1, 6)
    ext = [1, 536870910, 853144.6988601037]
    ret = []
    for i in range(10):
        if x[i][0] == -1:
            break

        pack = (x[i][0], x[i][1], x[i][2:])
        id, score, bbox = [k/v for k, v in zip(pack, ext)]
        if score < threshold:
            break

        bbox = bbox.clip(0, 512) / 512
        out_type = gdata.COCODetection.CLASSES[int(id)]
        print (i, out_type, int(id), score, bbox)
        ret.append((out_type, bbox[0:2], bbox[2:4]))
    return ret

models = {
    'mnist': {
        'path': '/data/std_out/cvm_mnist/',
        'input_size': 28,
        'transform': mnist_transform,
        'out_str': mnist_out,
    },
    'ssd_mobilenet_coco': {
        'path': '/data/std_out/ssd_512_mobilenet1.0_coco_tfm',
        'input_size': 512,
        'transform': ssd_transform,
        'out_str': ssd_out,
    }
}

model_name = 'ssd_mobilenet_coco'


img_mean = [0.485, 0.456, 0.406]
img_std  = [0.229, 0.224, 0.225]

input_size = models[model_name]['input_size']

def with_text(img, out_str, left_bottom, top_right):
    h, w, _ = img.shape
    left_bottom = [int(l*s) for s, l in zip(left_bottom, (w, h))]
    top_right = [int(l*s) for s, l in zip(top_right, (w, h))]
    point = (left_bottom[0], top_right[1] - 5)
    # print (left_bottom, top_right, img.shape)
    cv2.putText(img, out_str, point,
                cv2.FONT_ITALIC, text_font, (255, 0, 0), 2)

    cv2.rectangle(img,
                  tuple(left_bottom), tuple(top_right),
                  (0, 0, 255), 2)
    return img

model_root = os.path.join(models[model_name]["path"])
json, params = load_model(os.path.join(model_root, "symbol"),
                os.path.join(model_root, "params"))
net = CVMAPILoadModel(json, params)

def load_image():
    files = os.listdir(photo_dir)
    files = [os.path.join(photo_dir, f) for f in files]
    for f in os.listdir(photo_dir):
        fname = os.path.join(photo_dir, f)
        yield cv2.imread(fname)

if data_mode == "video_file":
    for f in os.listdir(photo_dir):
        org_frame = cv2.imread(os.path.join(photo_dir, f))
        frame = cv2.resize(org_frame, (input_size, input_size))
        frame = models[model_name]['transform'](frame)
        out = CVMAPIInference(net, img)
        out = np.array(out, np.float32)

        out_info = models[model_name]['out_str'](out)
        frame_with_res = org_frame
        for ostr, lb, tr in out_info:
            frame_with_res = with_text(frame_with_res, ostr, lb, tr)

        cv2.imwrite(os.path.join("./tests/python/photo_processed", f), frame_with_res)
elif data_mode == "videos":
    source = os.path.abspath(os.path.expanduser(source))
    file_name = os.path.basename(source)
    base_dir = os.path.dirname(source)

    vr = video.VideoReader(source)

    if dest is None:
        dest = os.path.join(base_dir, "processed_%s" % file_name)
    dest_dir = os.path.dirname(dest)

    num_count = 0
    is_ok, image = vr.read()
    w, h, _ = image.shape
    fout = video.VideoWriter(dest, dest_type)
    while True:
        if not is_ok:
            break

        if num_count % 10 == 0:
            print ("Processed: ", num_count)

        frame = cv2.resize(image, (input_size, input_size))
        #  image2 = frame[:]
        frame = models[model_name]['transform'](frame)
        out = CVMAPIInference(net, frame)
        out = np.array(out, np.float32)

        out_info = models[model_name]['out_str'](out)
        frame_with_res = image
        for ostr, lb, tr in out_info:
            frame_with_res = with_text(frame_with_res, ostr, lb, tr)

        if preview:
            cv2.imwrite(os.path.join(dest_dir, "tmp.jpg"), frame_with_res)
            cv2.imwrite(os.path.join(dest_dir, "origin.jpg"), image)
            exit()

        fout.write(frame_with_res)
        num_count += 1
        is_ok, image = vr.read()

    vr.release()
    fout.release()
    print ("Generating DONE!")
