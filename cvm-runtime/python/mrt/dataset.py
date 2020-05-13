import mxnet as mx
from mxnet import gluon
from mxnet import nd
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
import numpy as np
import requests
import tarfile

import os
from os import path
import math
import pickle
import logging

dataset_dir = path.expanduser("~/.mxnet/datasets")
src = "http://0.0.0.0:8827"

def extract_file(tar_path, target_path):
    tar = tarfile.open(tar_path, "r")
    if path.exists(path.join(target_path,
                tar.firstmember.name)):
        return
    tar.extractall(target_path)
    tar.close()

def download_files(category, files, base_url=src, root=dataset_dir):
    logger = logging.getLogger("dataset")
    root_dir = path.join(root, category)
    os.makedirs(root_dir, exist_ok=True)

    for df in files:
        url = path.join(base_url, 'datasets', category, df)
        fpath = path.join(root_dir, df)
        if path.exists(fpath):
            continue
        fdir = path.dirname(fpath)
        if not path.exists(fdir):
            os.makedirs(fdir)

        logger.info("Downloading dateset %s into %s from url[%s]",
                df, root_dir, url)
        r = requests.get(url)
        if r.status_code != 200:
            logger.error("Url response invalid status code: %s",
                    r.status_code)
            exit()
        r.raise_for_status()
        with open(fpath, "wb") as fout:
            fout.write(r.content)
    return root_dir

class Dataset:
    name = None

    def __init__(self, input_shape, base_url=src, root=dataset_dir, dataset_dir=None):
        self.ishape = input_shape

        self.root_dir = download_files(
            self.name, self.download_deps, base_url, root) \
            if dataset_dir is None else dataset_dir
        for fname in self.download_deps:
            if fname.endswith(".tar") or fname.endswith(".tar.gz"):
                extract_file(path.join(self.root_dir, fname), self.root_dir)

        self.data = None
        self._load_data()

    def metrics(self):
        pass

    def validate(self, metrics, predicts, labels):
        pass

    def _load_data(self):
        pass

    def __iter__(self):
        """ Returns (data, label) iterator """
        return iter(self.data)

    def iter_func(self):
        data_iter = iter(self)
        def _wrapper():
            return next(data_iter)
        return _wrapper

class COCODataset(Dataset):
    name = "coco"
    download_deps = ['val2017.zip']

    def _load_data(self):
        assert len(self.ishape) == 4
        N, C, H, W = self.ishape
        assert C == 3
        self.val_dataset = gdata.COCODetection(
            root=self.root_dir, splits='instances_val2017', skip_empty=False)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        self.data = gluon.data.DataLoader(
            self.val_dataset.transform(SSDDefaultValTransform(W, H)),
            batch_size=N, shuffle=False, batchify_fn=val_batchify_fn,
            last_batch='rollover', num_workers=30)

    def metrics(self):
        _, _, H, W = self.ishape
        metric = COCODetectionMetric(
            self.val_dataset, '_eval', cleanup=True, data_shape=(H, W))
        metric.reset()
        return metric

    def validate(self, metrics, predict, label):
        det_ids, det_scores, det_bboxes = [], [], []
        gt_ids, gt_bboxes, gt_difficults = [], [], []

        _, _, H, W = self.ishape
        assert H == W
        ids, scores, bboxes = predict
        det_ids.append(ids)
        det_scores.append(scores)
        # clip to image size
        det_bboxes.append(bboxes.clip(0, H))
        gt_ids.append(label.slice_axis(axis=-1, begin=4, end=5))
        gt_difficults.append(
            label.slice_axis(axis=-1, begin=5, end=6) \
            if label.shape[-1] > 5 else None)
        gt_bboxes.append(label.slice_axis(axis=-1, begin=0, end=4))

        metrics.update(det_bboxes, det_ids, det_scores,
                            gt_bboxes, gt_ids, gt_difficults)
        names, values = metrics.get()
        acc = {k:v for k,v in zip(names, values)}
        acc = float(acc['~~~~ MeanAP @ IoU=[0.50,0.95] ~~~~\n']) / 100
        return "{:6.2%}".format(acc)


class VOCDataset(Dataset):
    name = "voc"
    download_deps = ["VOCtest_06-Nov-2007.tar"]

    def _load_data(self):
        assert len(self.ishape) == 4
        N, C, H, W = self.ishape
        assert C == 3
        val_dataset = gdata.VOCDetection(
            root=path.join(self.root_dir, 'VOCdevkit'),
            splits=[('2007', 'test')])
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        self.data = gluon.data.DataLoader(
            val_dataset.transform(YOLO3DefaultValTransform(W, H)),
            N, False, batchify_fn=val_batchify_fn,
            last_batch='discard', num_workers=30)

    def metrics(self):
        metric = VOC07MApMetric(
            iou_thresh=0.5, class_names=gdata.VOCDetection.CLASSES)
        metric.reset()
        return metric

    def validate(self, metrics, predict, label):
        det_ids, det_scores, det_bboxes = [], [], []
        gt_ids, gt_bboxes, gt_difficults = [], [], []

        _, _, H, W = self.ishape
        assert H == W
        ids, scores, bboxes = predict
        det_ids.append(ids)
        det_scores.append(scores)
        # clip to image size
        det_bboxes.append(bboxes.clip(0, H))
        gt_ids.append(label.slice_axis(axis=-1, begin=4, end=5))
        gt_difficults.append(
            label.slice_axis(axis=-1, begin=5, end=6) \
            if label.shape[-1] > 5 else None)
        gt_bboxes.append(label.slice_axis(axis=-1, begin=0, end=4))

        metrics.update(det_bboxes, det_ids, det_scores,
                            gt_bboxes, gt_ids, gt_difficults)
        map_name, mean_ap = metrics.get()
        acc = {k:v for k,v in zip(map_name, mean_ap)}['mAP']
        return "{:6.2%}".format(acc)

class VisionDataset(Dataset):
    def metrics(self):
        return [mx.metric.Accuracy(),
                mx.metric.TopKAccuracy(5)]

    def validate(self, metrics, predict, label):
        metrics[0].update(label, predict)
        metrics[1].update(label, predict)
        _, top1 = metrics[0].get()
        _, top5 = metrics[1].get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

class ImageNetDataset(VisionDataset):
    name = "imagenet"
    download_deps = ["rec/val.rec", "rec/val.idx"]

    def _load_data(self):
        assert len(self.ishape) == 4
        N, C, H, W = self.ishape
        assert C == 3
        assert H == W

        crop_ratio = 0.875
        resize = int(math.ceil(H / crop_ratio))
        mean_rgb = [123.68, 116.779, 103.939]
        std_rgb = [58.393, 57.12, 57.375]
        rec_val = path.join(self.root_dir, self.download_deps[0])
        rec_val_idx = path.join(self.root_dir, self.download_deps[1])


        self.data = mx.io.ImageRecordIter(
            path_imgrec         = rec_val,
            path_imgidx         = rec_val_idx,
            preprocess_threads  = 24,
            shuffle             = False,
            batch_size          = N,

            resize              = resize,
            data_shape          = (3, H, W),
            mean_r              = mean_rgb[0],
            mean_g              = mean_rgb[1],
            mean_b              = mean_rgb[2],
            std_r               = std_rgb[0],
            std_g               = std_rgb[1],
            std_b               = std_rgb[2],
        )

    def __iter__(self):
        return self

    def __next__(self):
        data = self.data.next()
        return data.data[0], data.label[0]

class Cifar10Dataset(VisionDataset):
    name = "cifar10"
    download_deps = ["cifar-10-binary.tar.gz"]

    def _load_data(self):
        N, C, H, W = self.ishape
        assert C == 3 and H == W and H == 32
        transform_test = gluon.data.vision.transforms.Compose([
            gluon.data.vision.transforms.ToTensor(),
            gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                   [0.2023, 0.1994, 0.2010])])
        self.data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(root=self.root_dir,
                train=False).transform_first(transform_test),
            batch_size=N, shuffle=False, num_workers=4)


class QuickDrawDataset(VisionDataset):
    name = "quickdraw"

    def __init__(self, input_shape, is_train=False, **kwargs):
        self.download_deps = [
            "quickdraw_X.npy", "quickdraw_y.npy"] if is_train else \
            ["quickdraw_X_test.npy", "quickdraw_y_test.npy"]
        self.is_train = is_train
        super().__init__(input_shape, **kwargs)

    def _load_data(self):
        N, C, H, W = self.ishape
        assert C == 1 and H == 28 and W == 28
        X = nd.array(np.load(path.join(self.root_dir, self.download_deps[0])))
        Y = nd.array(np.load(path.join(self.root_dir, self.download_deps[1])))
        self.data = gluon.data.DataLoader(
                mx.gluon.data.dataset.ArrayDataset(X, Y),
                batch_size=N,
                last_batch='discard',
                shuffle=self.is_train,
                num_workers=4)

class MnistDataset(VisionDataset):
    name = "mnist"
    download_deps = ["t10k-images-idx3-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz",
                     "train-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz"]

    def _load_data(self):
        val_data = mx.gluon.data.vision.MNIST(
            root=self.root_dir, train=False).transform_first(data_xform)

        N, C, H, W = self.ishape
        assert C == 1 and H == 28 and W == 28
        self.data = mx.gluon.data.DataLoader(
            val_data, shuffle=False, batch_size=N)


class TrecDataset(Dataset):
    name = "trec"
    download_deps = ["TREC.train.pk", "TREC.test.pk"]

    def __init__(self, input_shape, is_train=False, **kwargs):
        self.is_train = is_train
        super().__init__(input_shape, **kwargs)

    def _load_data(self):
        fname = path.join(
            self.root_dir, self.download_deps[0] \
            if self.is_train else self.download_deps[1])

        # (38, batch), (batch,)
        with open(fname, "rb") as fin:
            self.data = pickle.load(fin)

        I, N = self.ishape
        assert I == 38

    def __iter__(self):
        data, label = [], []
        for x, y in self.data:
            if len(data) < self.ishape[1]:
                data.append(x)
                label.append(y)
            else:
                yield nd.transpose(nd.array(data)), nd.array(label)

                data, label = [], []

    def metrics(self):
        return {"acc": 0, "total": 0}

    def validate(self, metrics, predict, label):
        for idx in range(predict.shape[0]):
            res_label = predict[idx].asnumpy().argmax()
            data_label = label[idx].asnumpy()
            if res_label == data_label:
                metrics["acc"] += 1
            metrics["total"] += 1

        acc = 1. * metrics["acc"] / metrics["total"]
        return "{:6.2%}".format(acc)

DS_REG = {
    "voc": VOCDataset,
    "imagenet": ImageNetDataset,
    "cifar10": Cifar10Dataset,
    "quickdraw": QuickDrawDataset,
    "mnist": MnistDataset,
    "trec": TrecDataset,
    "coco": COCODataset,
}

# max value: 2.64
def load_voc(batch_size, input_size=416, **kwargs):
    fname = "VOCtest_06-Nov-2007.tar"
    root_dir = download_files("voc", [fname], **kwargs)

    extract_file(path.join(root_dir, fname), root_dir)
    width, height = input_size, input_size
    val_dataset = gdata.VOCDetection(root=path.join(root_dir, 'VOCdevkit'),
            splits=[('2007', 'test')])
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size,
        False,
        batchify_fn=val_batchify_fn,
        last_batch='keep',
        num_workers=30)
    return val_loader

def load_voc_metric():
    return VOC07MApMetric(iou_thresh=0.5, class_names=gdata.VOCDetection.CLASSES)

def load_imagenet_rec(batch_size, input_size=224, device_id=4, **kwargs):
    files = ["rec/val.rec", "rec/val.idx"]
    root_dir = download_files("imagenet", files, **kwargs)
    crop_ratio = 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    rec_val = path.join(root_dir, files[0])
    rec_val_idx = path.join(root_dir, files[1])

    val_data = mx.io.ImageRecordIter(
	path_imgrec         = rec_val,
	path_imgidx         = rec_val_idx,
	preprocess_threads  = 24,
	shuffle             = False,
	batch_size          = batch_size,

	resize              = resize,
	data_shape          = (3, input_size, input_size),
	mean_r              = mean_rgb[0],
	mean_g              = mean_rgb[1],
	mean_b              = mean_rgb[2],
	std_r               = std_rgb[0],
	std_g               = std_rgb[1],
	std_b               = std_rgb[2],

        device_id           = device_id,
    )
    return val_data

def load_cifar10(batch_size, input_size=224, num_workers=4, **kwargs):
    flist = ["cifar-10-binary.tar.gz"]
    root_dir = download_files("cifar10", flist, **kwargs)
    extract_file(path.join(root_dir, flist[0]), root_dir)
    transform_test = gluon.data.vision.transforms.Compose([
        gluon.data.vision.transforms.ToTensor(),
        gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                               [0.2023, 0.1994, 0.2010])])
    val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(root=root_dir,
                train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return val_data

def load_quickdraw10(batch_size, num_workers=4, is_train=False, **kwargs):
    files = ["quickdraw_X.npy", "quickdraw_y.npy"] if is_train else \
            ["quickdraw_X_test.npy", "quickdraw_y_test.npy"]
    root_dir = download_files("quickdraw", files, **kwargs)
    X = nd.array(np.load(path.join(root_dir, files[0])))
    y = nd.array(np.load(path.join(root_dir, files[1])))
    val_data = gluon.data.DataLoader(
            mx.gluon.data.dataset.ArrayDataset(X, y),
            batch_size=batch_size,
            last_batch='discard',
            shuffle=is_train,
            num_workers=num_workers)
    return val_data

def load_trec(batch_size, is_train=False, **kwargs):
    files = ["TREC.train.pk", "TREC.test.pk"]
    root_dir = download_files("trec", files, **kwargs)
    fname = path.join(root_dir, files[0] if is_train else files[1])
    with open(fname, "rb") as fin:
        dataset = pickle.load(fin)
        data, label = [], []
        for x, y in dataset:
            if len(data) < batch_size:
                data.append(x)
                label.append(y)
            else:
                yield nd.transpose(nd.array(data)), nd.transpose(nd.array(label))
                data, label = [], []

def load_mnist(batch_size, **kwargs):
    flist = ["t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz"]
    root_dir = download_files("mnist", flist, **kwargs)
    val_data = mx.gluon.data.vision.MNIST(root=root_dir, train=False).transform_first(data_xform)
    val_loader = mx.gluon.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)
    return val_loader

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255

def data_iter(dataset, batch_size, **kwargs):
    if dataset == "imagenet":
        data_iter = load_imagenet_rec(batch_size, **kwargs)
        def data_iter_func():
            data = data_iter.next()
            return data.data[0], data.label[0]
    elif dataset == "voc":
        val_data = load_voc(batch_size, **kwargs)
        data_iter = iter(val_data)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "trec":
        data_iter = load_trec(batch_size, **kwargs)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "mnist":
        val_loader = load_mnist(batch_size, **kwargs)
        data_iter = iter(val_loader)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "quickdraw":
        val_data = load_quickdraw10(batch_size, **kwargs)
        data_iter = iter(val_data)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "cifar10":
        val_data = load_cifar10(batch_size, **kwargs)
        data_iter = iter(val_data)
        def data_iter_func():
            return next(data_iter)
    else:
        assert False, "dataset:%s is not supported" % (dataset)
    return data_iter_func



