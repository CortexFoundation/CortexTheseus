
from cvm.base import CVMContext, CPU, GPU, FORMAL
from cvm.utils import load_model, load_np_data
from cvm.dll import CVMAPILoadModel, CVMAPIInference, CVMAPIGetOutputLength
from cvm import utils

import os
import sys
import argparse
import threading
import subprocess
import numpy as np
import cv2
import time


def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


    frame[:,:,0] = (frame[:,:,0] - 0.485)/0.229
    frame[:,:,1] = (frame[:,:,1] - 0.456)/0.224
    frame[:,:,2] = (frame[:,:,2] - 0.406)/0.225

CVMContext.set_global(GPU)
#cap = open_cam_onboard(1280, 720)
cap = open_cam_onboard(1920, 1080)

model = "ssd_512_mobilenet1.0_coco_tfm"
model = "resnet50_mxg"
model = "mobilenet1.0"

 ## loading model
if model == "ssd_512_mobilenet1.0_coco_tfm":
    model_root = os.path.join('/data/std_out',model)
    input_size = 512

    img_mean = [0.485, 0.456, 0.406]
    img_std  = [0.229, 0.224, 0.225]

    label_file = "label_coco.txt"
    label = []
    f = open(label_file,'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        label.append(line)

    #print(label)
else:
    model_root = os.path.join('/data/std_out',model)
    input_size = 224

    img_mean = [0.485, 0.456, 0.406]
    img_std  = [0.229, 0.224, 0.225]

    label_file = "label_imagenet.txt"
    label = []
    f = open(label_file,'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        label.append(line.split('--')[-1])


print("model_root === ", model_root)

json, params = load_model(os.path.join(model_root, "symbol"),
                 os.path.join(model_root, "params"))




net = CVMAPILoadModel(json, params)
#print("output len net: ", CVMAPIGetOutputLength(net))  # 2400

#data_path = os.path.join(model_root, "data.npy")
#data = load_np_data(data_path)
#print(data.shape())

frame_id = 0

while(1):

    t0 = time.time()
    ## data preprocessing
    ret, frame = cap.read()
    #print(ret, frame.shape)
    #cv2.imshow('',frame)
 
    frame_id += 1

    if frame_id % 30 > 0:
        continue
    else:
        frame_id = 0


    if frame.shape[0] > frame.shape[1]:
        square_size = frame.shape[1]
        start_p = (int)((frame.shape[0] - frame.shape[1])/2)
        frame = frame[start_p:start_p+frame.shape[1],:,:]
    else:
        square_size = frame.shape[0]
        start_p = (int)((frame.shape[1] - frame.shape[0])/2)
        frame = frame[:,start_p:start_p+frame.shape[0],:]

    frame = cv2.resize(frame, (input_size,input_size))
    frame0 = frame
    
    frame = frame[:,:,::-1].astype(np.float32)/255.0


    frame[:,:,0] = (frame[:,:,0] - img_mean[0])/img_std[0]
    frame[:,:,1] = (frame[:,:,1] - img_mean[1])/img_std[1]
    frame[:,:,2] = (frame[:,:,2] - img_mean[2])/img_std[2]


    frame = frame.swapaxes(0,2)
    frame = frame.swapaxes(1,2)
    frame = frame[np.newaxis,:]


    frame = np.clip(frame*48.10606060,-127,127).astype(np.int8).tobytes()
    #print("after", frame[0,0,0],frame[0,0,1],frame[0,0,2])
    #print(np.max(frame), np.min(frame))
    t1 = time.time()
    print("time for image capture and preprocessing: %s"%(t1-t0))
    

    out = CVMAPIInference(net, frame)
    t2 = time.time()
    print("time for CVMAPIInference: %s"%(t2-t1))


    if model == "ssd_512_mobilenet1.0_coco_tfm":
        out = np.array(out, np.float32).reshape((-1,6))

        out_float = out/[1, 536870910, 853144.6988601037, 853144.6988601037, 853144.6988601037, 853144.6988601037]
        box = []
        for i in range(out_float.shape[0]):
            if out_float[i][1] < 0.6:
                continue

            p1 = (int(out_float[i][2]), int(out_float[i][3]))
            p2 = (int(out_float[i][4]), int(out_float[i][5]))

            p3 = (max(p1[0], 15), max(p1[0], 15))
            title = "label:{} conf:{}".format(label[int(out_float[i][0])], out_float[i][1])

            cv2.rectangle(frame0, p1, p2, (0, 255, 0))
            cv2.putText(frame0, title, p3, cv2.FONT_ITALIC, 0.3, (255,0,0),1)
    else:
        max_ind = np.argmax(np.array(out, "float32"))
        cv2.putText(frame0, label[max_ind], (20,20), cv2.FONT_ITALIC, 0.3, (255,0,0),1)

    cv2.imshow('img',cv2.resize(frame0,(1024, 1024)))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break;
   

cap.release()
cv2.destroyAllWindows()

