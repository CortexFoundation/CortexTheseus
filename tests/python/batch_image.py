from cvm.base import CVMContext, CPU, GPU, FORMAL

from cvm.utils import load_model, load_np_data
from cvm.dll import CVMAPILoadModel, CVMAPIInference, CVMAPIGetOutputLength
from cvm import utils
import json

import os
import sys
import argparse
import threading
import subprocess
import numpy as np
import cv2
import time

CVMContext.set_global(GPU)


# input options
json_file = "./tests/python/conf.json"
x = open(json_file, encoding="utf-8")
y = x.read()
configure = json.loads(y)


Cython = configure["Cython"]
model_id = configure["model_id"]
data_mode = configure["data"][str(configure["data_select"])]
threshold = configure["threshold"]
disp_mode = configure['display'][str(configure["disp_mode"])]
font_mode = configure['font'][str(configure["font_mode"])]
video_fps = configure['video_fps']
frame_len = configure['frame_len'] # in second


if font_mode == "small":
    text_font = 1
else:
    text_font = 2

models = [[0,'/data/std_out/ssd','ssd'], [1,'/data/std_out/shufflenet','shuffle'], [2,'/data/std_out/ssd_512_mobilenet1.0_coco_tfm','ssdmobilenet'], [3, "/data/std_out/resnet50_mxg",'resnet50'], [4, "/data/std_out/mobilenet1.0",'mobilenet']]

img_mean = [0.485, 0.456, 0.406]
img_std  = [0.229, 0.224, 0.225]

if model_id in {1, 3, 4}:
    input_size = 224
elif model_id in {0, 2}:
    input_size = 512
else:
    input_size = 416

if model_id in {2,}:
    label_file = "./tests/python/label_coco.txt"
else:
    label_file = "./tests/python/label_imagenet.txt"
photo_dir = "./tests/python/photo/"

def preprocess(img, img_mean, img_std, netin_size):
    print("img.shape = ", img[0,0,0])
    if img.shape[0] > img.shape[1]:
        #square_size = img.shape[1]
        start_p = (int)((img.shape[0] - img.shape[1])/2)
        img = img[start_p:start_p+img.shape[1],:,:]
    else:
        #square_size = img.shape[0]
        start_p = (int)((img.shape[1] - img.shape[0])/2)
        img = img[:,start_p:start_p+img.shape[0],:]

    img = cv2.resize(img, (netin_size,netin_size))
    img = img[:,:,::-1].astype(np.float32)/255.0
    
    for i in range(3):
        img[:,:,i] = (img[:,:,i] - img_mean[i])/img_std[i]

    img = img.swapaxes(0,2)
    img = img.swapaxes(1,2)
    img = img[np.newaxis,:]
    img = np.clip(img*48.10606060,-127,127).astype(np.int8).tobytes()
    return img

def get_label(model_id):
    if model_id in {2, }:
        label_file = "./tests/python/label_coco.txt"
        label = []
        f = open(label_file,'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label.append(line)
    else:
        label_file = "./tests/python/label_imagenet.txt"
        label = []
        f = open(label_file,'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label.append(line.split('--')[-1])
    return label

def get_net(model_id, Cython):
    model_root = os.path.join(models[model_id][1])
    #print("model_root === ", model_root)
    if Cython == 1:
        device_id = 0
        json, params = utils.load_model(
            os.path.join(model_root, "symbol"),
            os.path.join(model_root, "params"))
        from cvm.dll import libcvm
        lib = libcvm.CVM(json, params, device_id)
        net = []
    else:
        json, params = load_model(os.path.join(model_root, "symbol"),
                        os.path.join(model_root, "params"))
        net = CVMAPILoadModel(json, params)
        lib = []
    #print("output len net: ", CVMAPIGetOutputLength(net))
    return net, lib

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

def infer(net, lib, img):
    if net:
        out = CVMAPIInference(net, img)
    else:
        out = lib.Inference(img)
    out = np.array(out, np.float32)
    return out

def gen_text(out_norm, img0, label, scaler):
    
    def putText(img, width, height, label, point, conf):

        if disp_mode == "original":
            scaler1 = scaler
        else:
            scaler1 = 2*scaler


        if conf:
            cv2.putText(img, "{:.2f}".format(conf), point, cv2.FONT_ITALIC, text_font, (255,0,0), 2)
            point = (point[0], int(point[1]+10*scaler1*text_font))
        
        text = []
        one_line = ''
        for word in set(label):
            one_line = one_line + word + ' '
            if len(one_line)*12*5 > width/text_font*2:
                text.append(one_line)
                one_line = ''
        text.append(one_line)
        text = list(set(text))

        for i in range(len(text)):
            cv2.putText(img, text[i], (point[0], point[1]+int(i*25*scaler1/2*text_font)), cv2.FONT_ITALIC, text_font, (255,0,0), 2)
        return img

    if len(out_norm.shape) == 1:
        max_ind = np.argmax(np.array(out_norm, "float32"))
        print("max_ind = ", max_ind)
        label0 = label[max_ind].strip("\'").strip("\"").strip("\n").split()

        p3 = (int(20*scaler),int(20*scaler))
        putText(img0, img0.shape[1], img0.shape[0], label0, p3, [])
    else:
        for i in range(out_norm.shape[0]):
            if out_norm[i][1] < max(0.0001, threshold):
                continue
            label0 = label[int(out_norm[i][0])].strip("\'").strip("\"").strip("\n").split()

            p1 = (int(out_norm[i][2]*scaler), int(out_norm[i][3]*scaler))
            p2 = (int(out_norm[i][4]*scaler), int(out_norm[i][5]*scaler))
            p3 = (max(int(p1[0]), int(15*scaler)), max(int(p1[1]), int(15*scaler)))

            img0 = putText(img0,p2[0]-p1[0], p2[1]-p1[1], label0, p3, out_norm[i][1])
            cv2.rectangle(img0, p1, p2, (0, 0, 255), 2)
    return img0



net_o, lib_o = get_net(model_id, Cython)
label = get_label(model_id)

if data_mode == "video_file":
    for f in os.listdir(photo_dir):
        frame = cv2.imread(os.path.join(photo_dir, f))
        frame1 = preprocess(frame, img_mean, img_std, input_size)
        net_out = infer(net_o, lib_o, frame1)

        if net_out.shape[0] == 600:
            out_norm = net_out.reshape((-1,6))
            out_norm = out_norm/[1, 536870910, 853144.6988601037, 853144.6988601037, 853144.6988601037, 853144.6988601037]
        else:
            out_norm = net_out

        if disp_mode == "original":
            scaler = frame.shape[0]/input_size
            img0 = frame
        else:
            scaler = 1
            img0 = cv2.resize(frame, (input_size, input_size))

        frame_text = gen_text(out_norm, img0, label, scaler)

        cv2.imwrite(os.path.join("./tests/python/photo_processed", f), frame_text)
        cv2.imshow('',frame_text)
        cv2.waitKey(1)

    mk_video = "ffmpeg -i ./tests/python/photo_processed/test_%5d.jpg ./tests/python/video_{}.mp4".format(models[model_id][2])
    os.system(mk_video)
else:
    cap = open_cam_onboard(1280, 720)
    video_id = 0
    frame_id = 0
    start_time = time.time()
    while(1):
        #print("frame_id = ", frame_id)
        _, frame = cap.read()
        cv2.flip(frame,0,frame )
        print(frame.shape)
        h, w = frame.shape[0], frame.shape[1]
        frame = frame[:, (w-h)//2:(w-h)//2+h,:]
        print("frame[0,0,0] = ", frame[0,0,0])
        frame1 = preprocess(frame, img_mean, img_std, input_size)
        net_out = infer(net_o, lib_o, frame1)

        if net_out.shape[0] == 600:
            out_norm = net_out.reshape((-1,6))
            out_norm = out_norm/[1, 536870910, 853144.6988601037, 853144.6988601037, 853144.6988601037, 853144.6988601037]
        else:
            out_norm = net_out
            print("net_out[0:10] = ",net_out[0:10])

        if disp_mode == "original":
            scaler = frame.shape[0]/input_size
            img0 = frame
        else:
            scaler = 1
            img0 = cv2.resize(frame, (input_size, input_size))

        frame_text = gen_text(out_norm, img0, label, scaler)
        real_frame_id = int((time.time() - start_time)*video_fps)

        print("video_id = ", video_id, " real_frame_id = ",real_frame_id, " frame_id = ", frame_id)

        if real_frame_id > frame_id:
            pad_num = real_frame_id-frame_id
            for pad_id in range(pad_num):
                frame_id = frame_id + 1
                cv2.imwrite(os.path.join("./tests/python/photo_processed", "{:0>5d}.jpg".format(frame_id)), frame_text)

        #cv2.imshow('',frame_text)
        #cv2.waitKey(1)

        if frame_id >= int(frame_len*video_fps):
            mk_video = "ffmpeg -y -r {} -i ./tests/python/photo_processed/%5d.jpg ./tests/python/cap_{}_{}.mp4".format(video_fps, models[model_id][2], video_id)
            os.system(mk_video)
            os.system("rm ./tests/python/photo_processed/*.*")
            start_time = time.time()
            frame_id = 0
            video_id = video_id + 1





#cap = open_cam_onboard(1280, 720)
#cap = open_cam_onboard(1920, 1080)
