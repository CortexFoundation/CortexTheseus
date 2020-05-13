## MRT Installation

#### 1. Install Python

#### 2. MRT Preparation

2.1. Clone the project

```
git clone git@github.com:CortexFoundation/cvm-runtime.git
```



2.2. Build the Shared Library

```
make -j8
```



2.3. Install other dependancies

​	First, create the file `requirements.txt` to see which dependancies are to be installed. 

​	For cuda with version 9.2, the default file looks like this:

>mxnet-cu92mkl
>
>gluoncv
>
>decorator

​	If other packages are also needed while testing, append the package name at a newline.

​	Use the following command to install all the dependancies.

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<200b>  An alternative command is available.

```
pip install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/web/simple
```



2.4. Create a Data Folder

```
mkdir data
```



#### 3. MRT Pipeline

3.1. Download (Customize) dataset

The source files of MRT validation needed for the currently mainstream datasets and corresponding dataset label file (under `path/to/cvm-runtime/docs/mrt`) are:

| Dataset Name | Source Files                                                 | Dataset Labels      |
| ------------ | ------------------------------------------------------------ | ------------------- |
| coco         | val2017.zip                                                  |                     |
| voc          | VOCtest_06-Nov-2007.tar                                      | voc_labels.txt      |
| imagenet     | rec/val.rec<br />rec/val.idx                                 | imagenet_labels.txt |
| cifar10      | cifar-10-binary.tar.gz                                       |                     |
| quickdraw    | quickdraw_X.npy<br />quickdraw_y.npy (is_train=True)<br />**or**<br />quickdraw_X_test.npy<br />quickdraw_y_test.npy (is_train=False) |                     |
| mnist        | t10k-images-idx3-ubyte.gz<br />t10k-labels-idx1-ubyte.gz<br />train-images-idx3-ubyte.gz<br />train-labels-idx1-ubyte.gz |                     |
| trec         | TREC.train.pk<br />TREC.test.pk                              |                     |

Or download other custom dataset if needed.



3.2. Pre-process Datasets

Please refer to https://gluon-cv.mxnet.io/build/examples_datasets/index.html for reference.

For example, dataset records is needed for`imagenet` datasets.

run `im2rec.py` as described in https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html#sphx-glr-build-examples-datasets-recordio-py.



3.3. Predefined Models

Download predefined gluonzoo models, please refer to https://gluon-cv.mxnet.io/model_zoo/index.html for reference.

The following models has been successfully tested in MRT:

```
resnet50_v1
resnet50_v2
resnet18_v1
resnet18v1_b_0.89
qd10_resnetv1_20
densenet161
alexnet
cifar_resnet20_v1
mobilenet1_0
mobilenetv2_1.0
shufflenet_v1
squeezenet1.0
vgg19
trec
mnist
yolo3_darknet53_voc
yolo3_mobilenet1.0_voc
ssd_512_resnet50_v1_voc
ssd_512_mobilenet1.0_voc
```

Or convert other (customized) models into mxnet models if needed.



3.4. Configure model

Create `<your_model_name>.ini` in `path/to/cvm-runtime/python/mrt/model_zoo`, please refer to  https://github.com/CortexFoundation/cvm-runtime/blob/ryt_tune/python/mrt/model_zoo/config.example.ini for sample configuration.



3.5. Run MRT

 MRT includes pre-process stages, quantization stage and post process stages, including `preparation`, `split model`, `calibration`, `quantization`, `merge model`, `evaluation` and `compilation`. The main work done in each stage is:

| Stage        | Main Work                                                    |
| ------------ | ------------------------------------------------------------ |
| prepare      | Model initializtion: duplicate name check, attach input shape, fuse multiple inputs, input name replacement and validate attributes of operators, fuse multiple outputs, fuse constant, fuse transpose, equivalent operator transformation and get unique parameters. |
| split model  | Split the given model with respect to the given operator names (detections models only). |
| calibration  | Calibrate the current model after setting mrt data. See https://github.com/CortexFoundation/cvm-runtime/blob/ryt_tune/docs/mrt/mrt.md for related APIs. |
| quantization | Quantize the current model after calibration. See https://github.com/CortexFoundation/cvm-runtime/blob/ryt_tune/docs/mrt/mrt.md for related APIs. |
| merge model  | Merge the split models with respect to the given operator names (detections models only). |
| evaluation   | (Optional stage) Compare the results between the predefined model and the quantized model based on the given dataset. See https://github.com/CortexFoundation/cvm-runtime/blob/ryt_tune/docs/mrt/mrt.md for model accuracy tested. |
| compilation  | (Optional stage) Compile the quantized graph into cvm graph and dump the graph define, parameters, pre-quantize model ext and unit-batch test data. |

Please run the following command under `path/to/cvm-runtime/` to execute the MRT:

```bash
python python/mrt/main2.py python/mrt/model_zoo/<your_model_name>.ini
```



## Reference

https://pypi.org/project/conda/#files

https://mxnet.apache.org/versions/master/install/index.html?platform=Linux&language=Python&processor=GPU
