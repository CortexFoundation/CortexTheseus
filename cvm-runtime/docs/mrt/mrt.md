# Model Representation Tool Documentation

[TOC]

## Introdution

MRT, short for **Model Representation Tool**, aims to convert floating model into a deterministic and non-data-overflow network. MRT links the off-chain developer community to the on-chain ecosystem, from Off-chain deep learning to MRT transformations, and then uploading to Cortex Blockchain for on-chain deterministic inference.

A full deterministic deep learning framework designed by Cortex is ran within CVM, the Cortex Virtual Machine ,and the integral part in our Cortex Blockchain source is called CVM runtime. All data flow in CVM is an integer with some precision ranged in 0 and 32. We proposed approaches to certify the data non-flow over INT32. The model that goes under MRT transformation can be accepted by CVM, which we called it on-chain model.

MRT is based on the MXNet symbol, doing operations on the whole operators with topological order in models. Besides, for scalability, we've researched the model transformation from TensorFlow into MXNet, models such as mobilenet, inception_v3 have been successfully converted and more operators will be supported in the future. Other deep learning frameworks like PyTorch and Caffe is in the roadmap of our plan.

MRT transformation usage is simple to model-training programmer since we have separated model quantization procedures from source code. One can invoke MRT via programming or configuring the settings file, more detail usage is introduced as below.

## Configuration File 

MRT has separated model quantization configurations from source code for simplifying the user-usage. So one can quantize their model quickly via configuring the .ini file. The running command script is as below.

``` bash
python cvm/quantization/main2.py config/file/path
```

Please refer to the example file: cvm/quantization/models/config.example.ini [link](https://github.com/CortexFoundation/tvm-cvm/blob/wlt/cvm/models/config.example.ini) for more configuration details. Copy the example file and configure the model's quantization settings locally. We have quantized and tested accuracy for some available models in MXNet gluon zoo with configurations file, whose settings are located in [link](https://github.com/CortexFoundation/tvm-cvm/blob/wlt/cvm/models/) cvm/quantization/models for reference. These accuracies are organized into a chart for analysis in section [Model Testing](#Model Testing).

The unify quantization procedure is defined in file: cvm/quantization/main2.py, refer to [main2](https://github.com/CortexFoundation/tvm-cvm/blob/ryt_tmp/cvm/quantization/main2.py) for more quantization details.

## Developer API

The Main public quantization API is located at cvm/quantization/transformer.py, see the detail interface in the following sections. And the main quantization procedure is: 

    Model Load >>> Preparation >>> [Optional] Model Split >>>
    
    Calibration >>> Quantization >>> [Optional] Model Merge >>> Compilation to CVM,

which maps the class methods: 

    Model.load >>> Model.prepare >>> [Optional] Model.split >>> 
    
    MRT.calibrate >>> MRT.quantize >>> [Optional] ModelMerger.merge >>> Model.to_cvm.

The Calibration and Quantization pass is achieved in class MRT.

### Split & Merge

MRT supports for most of MXNet operators while there still exists some unsupported. We advise splitting the model into two sub-graph if there are some unsupported operators and only quantizing the half model (named base_model, indicating the input nodes to split operators generally). In other words, it's the user's responsibility to select the split keys of splitting the original model, while the half model is ignored to quantization pass if necessary. 

#### Currently Supported Operators

Below operators are carefully selected by the MRT developers. The unsupported oprators are the ones that are unquantifiable. For the unsupported operators, you can either split the model with disable-quantization attributes or contact the MRT developers through GitHub for assitance.

##### Transformer

| Operator     | Supported          | Operator    | Supported          |
| ------------ | ------------------ | ----------- | ------------------ |
| SliceAxis    | :heavy_check_mark: | SwapAxis    | :heavy_check_mark: |
| Slice        | :heavy_check_mark: | Flatten     | :heavy_check_mark: |
| SliceLike    | :heavy_check_mark: | Concat      | :heavy_check_mark: |
| Transpose    | :heavy_check_mark: | where       | :heavy_check_mark: |
| repeat       | :heavy_check_mark: | expand_dims | :heavy_check_mark: |
| SliceChannel | :heavy_check_mark: | tile        | :heavy_check_mark: |
| squeeze      | :heavy_check_mark: | Reshape     | :heavy_check_mark: |
| clip         | :heavy_check_mark: | Embedding   | :heavy_check_mark: |

##### NN

| Operator       | Supported          | Operator | Supported          |
| -------------- | ------------------ | -------- | ------------------ |
| Convolution    | :heavy_check_mark: | Pad      | :heavy_check_mark: |
| FullyConnected | :heavy_check_mark: | relu     | :heavy_check_mark: |
| LeakyReLU      | :heavy_check_mark: | Pooling  | :heavy_check_mark: |
| UpSampling     | :x:(TODO)          | softmax  | :heavy_check_mark: |
| BatchNorm      | :heavy_check_mark: | Dropout  | :heavy_check_mark: |
| Activation     | :heavy_check_mark: |          |                    |

##### Broadcast

| Operator      | Supported          | Operator          | Supported          |
| ------------- | ------------------ | ----------------- | ------------------ |
| broadcast_div | :x:                | broadcast_add     | :heavy_check_mark: |
| broadcast_sub | :heavy_check_mark: | broadcast_mul     | :heavy_check_mark: |
| broadcast_to  | :x:                | broadcast_greater | :x:                |

##### Elemwise

| Operator        | Supported          | Operator     | Supported          |
| --------------- | ------------------ | ------------ | ------------------ |
| _mul_scalar     | :heavy_check_mark: | _div_scalar  | :heavy_check_mark: |
| elemwise_add    | :heavy_check_mark: | elemwise_sub | :heavy_check_mark: |
| ceil            | :x:                | round        | :x:                |
| fix             | :x:                | floor        | :x:                |
| abs             | :x:                | sigmoid      | :heavy_check_mark: |
| exp             | :heavy_check_mark: | negative     | ✔️                  |
| _minimum        | :x:                | _maximum     | :x:                |
| _plus_scalar    | :heavy_check_mark: | zeros_like   | :heavy_check_mark: |
| _greater_scalar | :heavy_check_mark: | ones_like    | ✔️                  |

##### Reduce

| Operator | Supported          | Operator | Supported |
| -------- | ------------------ | -------- | --------- |
| max      | :heavy_check_mark: | min      | :x:       |
| sum      | :heavy_check_mark: | argmin   | :x:       |
| argmax   | :x:                |          |           |

##### Vision

| Operator         | Supported | Operator | Supported |
| ---------------- | --------- | -------- | --------- |
| _contrib_box_nms | :x:       |          |           |

##### Others

| Operator | Supported          | Operator | Supported |
| -------- | ------------------ | -------- | --------- |
| _arange  | :heavy_check_mark: | Custom   | ❌         |
### Public Interface

#### Model

A wrapper class for MXNet symbol and params which indicates model. All the quantization passes return the class instance for unify representation. Besides, the class has wrapped some user-friendly functions API  listed as below.

| func name                                          | usage                                                        |
| -------------------------------------------------- | ------------------------------------------------------------ |
| input_names()                                      | List the model's input names.                                |
| output_names()/names()                             | List the model's output names.                               |
| to_graph([dtype, ctx])                             | A convenient method to create model runtime.<br />Returns mxnet.gluon.nn.SymbolBlock. |
| save(symbol_file, params_file)                     | Dump model to disk.                                          |
| load(symbol_file, params_file)                     | **[staticmethod]** Load model from disk.                     |
| split(keys)                                        | Split the model by `keys` of model internal names.<br />Returns two sub-graph Model instances. |
| merger(base, top[, base_name_maps])                | [**staticmethod**] Returns the ModelMerger with two Model instance. |
| prepare([input_shape])                             | Model preparation passes, do operator checks, operator fusing, operator rewrite, ...etc. |
| to_cvm(model_name[, datadir, input_shape, target]) | Compile current mxnet quantization model into CVM accepted JSON&BINARY format. |

#### MRT

A wrapper class for model transformation tool which simulates deep learning network integer computation within a float-point context. Model calibration and quantization are performed based on a specified model. This class has wrapped some user-friendly functions API introduced as below.

| func name                        | usage                                                        |
| -------------------------------- | ------------------------------------------------------------ |
| set_data(data)                   | Set the data before calibration.                             |
| calibrate([ctx, lambd, old_ths]) | Calibrate the current model after setting mrt data.<br />Context on which intermediate result would be stored, hyperparameter lambd and reference threshold dict could also be specified. <br />Return the threshold dict of node-level output. |
| set_threshold(name, threshold)   | Manually set the threshold of the node output, given node name. |
| set_th_dict(th_dict)             | Manually set the threshold dict.                             |
| set_input_prec(prec)             | Set the input precision before quantization.                 |
| set_out_prec(prec)               | Set the output precision before quantization.                |
| set_softmax_lambd(val)           | Set the hyperparameter softmax_lambd before quantization.    |
| set_shift_bits(val)              | Set the hyperparameter shift_bits before quantization.       |
| quantize()                       | Quantize the current model after calibration.<br />Return the quantized model. |
| get_output_scales()              | Get the output scale of the model after quantization.        |
| get_maps()                       | Get the current name to old name map of the outputs after calibration or quantization. |
| get_inputs_ext()                 | Get the input_ext of the input after quantization.           |
| save(model_name[, datadir])      | save the current mrt instance into disk.                     |
| load(model_name[, datadir])      | [**staticmethod**]Return the mrt instance.<br />The given path should contain corresponding '.json' and '.params' file storing model information and '.ext' file storing mrt information. |

#### ModelMerger

A wrapper class for model merge tool. This class has wrapped some user-friendly functions API introduced as below.

| func name                             | usage                                                        |
| ------------------------------------- | ------------------------------------------------------------ |
| merge([callback])                     | Return the merged model. <br />Callback function could also be specified for updating the top node attributes. |
| get_output_scales(base_oscales, maps) | Get the model output scales after merge.<br />Base model output scales and base name maps should be specified. |

## Model Testing

Some models have been tested and the precision before and after quantization is provided in the following chart for reference.

| model name               | Iteration | evalfunc                     | quantize                     | total sample |
| ------------------------ | --------- | ---------------------------- | ---------------------------- | ------------ |
| resnet50_v1              | 312       | top1=77.39%<br />top5=93.59% | top1=76.47%<br />top5=93.28% | 50080        |
| resnet50_v2              | 312       | top1=77.15%<br />top5=93.44% | top1=70.76%<br />top5=89.56% | 50080        |
| resnet18_v1              | 312       | top1=70.96%<br />top5=89.93% | top1=70.11%<br />top5=89.60% | 50080        |
| resnet18v1_b_0.89        | 312       | top1=67.21%<br />top5=87.45% | top1=63.75%<br />top5=85.63% | 50080        |
| quickdraw_wlt            | 349       | top1=81.90%<br />top5=98.26% | top1=81.83%<br />top5=98.24% | 56000        |
| qd10_resnetv1_20         | 349       | top1=85.79%<br />top5=98.73% | top1=85.79%<br />top5=98.73% | 56000        |
| densenet161              | 312       | top1=77.62%<br />top5=93.82% | top1=77.32%<br />top5=93.63% | 50080        |
| alexnet                  | 312       | top1=55.91%<br />top5=78.75% | top1=51.69%<br />top5=77.99% | 50080        |
| cifar_resnet20_v1        | 62        | top1=92.88%<br />top5=99.78% | top1=92.82%<br />top5=99.75% | 10000        |
| mobilenet1_0             | 312       | top1=70.77%<br />top5=89.97% | top1=66.11%<br />top5=87.35% | 50080        |
| mobilenetv2_1.0          | 312       | top1=71.51%<br />top5=90.10% | top1=69.39%<br />top5=89.30% | 50080        |
| shufflenet_v1            | 312       | top1=63.48%<br />top5=85.12% | top1=60.45%<br />top5=82.95% | 50080        |
| squeezenet1.0            | 312       | top1=57.20%<br />top5=80.04% | top1=55.16%<br />top5=78.67% | 50080        |
| tf_inception_v3          | 312       | top1=55.58%<br />top5=77.56% | top1=55.54%<br />top5=83.03% | 50080        |
| vgg19                    | 312       | top1=74.14%<br />top5=91.78% | top1=73.75%<br />top5=91.67% | 50080        |
| trec                     | 28        | 97.84%                       | 97.63%                       | 1102         |
| yolo3_darknet53_voc      | 29        | 81.37%                       | 82.08%                       | 4800         |
| yolo3_mobilenet1.0_voc   | 29        | 75.98%                       | 71.53%                       | 4800         |
| ssd_512_resnet50_v1_voc  | 29        | 80.27%                       | 80.01%                       | 4800         |
| ssd_512_mobilenet1.0_voc | 29        | 75.57%                       | 71.32%                       | 4800         |
| mnist                    | 62        | top1=99.18%<br />top5=100%   | 99.17%<br />top5=100%        | 10000        |