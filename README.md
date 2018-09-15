## Cortex AI Consensus Specifications V0.1 (For Cortex Test Net)

### 1. Abstract

#### 1.1 Main Design: AI Consensus

AI Inference quantization solution given by Google, Nvidia, Intel,  Apple and Amazon are inspired by computation acceleration, while AI consensus usually require a setup one step further: with same input and same AI computation graph, the final computation result must be bitwise equal, i.e. "deterministicity". To implement "All-Integer" inference data pipeline, only integer tensor/matrix multiplication is far from enough.

 We noticed the necessity of AI consensus since Nov. 2017, and implement the whole data pipeline on quantized AI computation on integers by September. 2018, which means we also implement the core functionalities without depending on any existing unopen-sourced complicated toolchains.

#### 1.2 Compatibility Design: Highest level of Hardware Compatibility

All specifications and designs of Cortex AI Consensus will not be obstacles when implementing AI inference on OpenCL devices, Smartphone ARM chips, and FPGAs, but our initial version is not optimized for certain kinds of hardware.

No float number multiplications are involved. 

Certainly kinds of integer multiplications are simplified to bit shifting without loss of deterministicity.

### 2. Supported AI Model Specifications

Cortex Testnet have already support AI models with INT8 with INT32 runtime features. Compared to FP32 models, AI model size reduced by 75% of original model, but runtimememory consumption remains the same. On the same Cortex node which can execute AI model inference, the number of models that can be cached is 4 times as before, but queueing strategy of data to be inferenced is not optimized.

Cortex Testnet support following sort of neural network layers:

#### 2.1 Convolutional Layers

Convolutional layers support INT8 parameters.

Conv1D , Conv2D of any kernel sizes, dilation sizes and padding sizes are supported.

Conv3D is not supported in TestNet, and will be supported in Cortex MainNet.

For Conv1D() with kernel size (k), please use Conv2D with kernel size (1,k).

For Conv2D() with dilation, please use Conv2D with larger convolution kernels.

#### 2.2 Normalization Layers

Normalization parameters for channels is supported, which is by default supporting Batch Normalization.

Cortex TestNet support integer parameter of Normalization.

Cortex TestNet recommend developers constrain all the scale factors to the form of  2^K,  when K is a negative integer.

#### 2.3 Non-Linear Activation Layers

ReLu is supported.

For PReLu, please add up two ReLu with integer parameter p.

We recommend using p in the form of  2^K,  when K is a negative integer.

#### 2.4 Fully Connected Layers

Only INT32 AI runtime features is supported: It accepts INT8 tensors, matrices and vectors, and returns INT32 result. 

#### 2.5 Residual Network Configuration

We refer to multiple implementations of residual-like setups:

 * Highway network brings out the initial  idea of  x + cf(x) 
 * Residual Network implemented c=1
 * Google implemented  Inception-ResNetv2 with c=0.2, 0.15, 0.1 etc.

Now c is fixed to 1.

We will generalize c to the form of 2^K.  K is an integer.

For each layer of runtime features,  we will record the overall shiftbit of this whole layer instead recording these values in floating point numbers. 

Adjustment of this shift is recorded when residual operation is performed.

#### 2.6 Total Model Setup

Input images channel must be compatible with the Inference model.

We’ve tackled the data overflow issue and OOM issue. 

We recommend input images less than 3 * 224 * 224, a model less than or equal to 200 layers, and channel size less than or equal to 2048.

#### 2.7 Addition

Please use Conv1D and quantization instead of RNN-like structures, e.g. GRU/LSTM.  Total Accuracy will not be harmed in NLP/OCR practices.


### 3. Further Development

#### 3.1 On Supporting INT2/INT4 AI models

By the end of Sep. 2018:

 * Apple is developing INT4 AI model inference.
 * Huawei had released INT8 AI model inference on their newly released Kylin 980.
 * Nvidia will release INT4 AI model inference on their newly released Turing GPU cards like GeForce 2080Ti.

We will support INT4 AI model inference development before Cortex Mainnet online, and continuously research on the low bitwidth AI model’s parameters.

#### 3.2 On Supporting INT8/INT16 AI runtime features

By the end of Sep. 2018:
 * Google, Intel and Intel  had implemented AI model inference with FP16 runtime features.
 * Google released their experiment result on  INT8 model with INT16 activation, without loss of AI model accuracy.
 * No existing sources showing that any corporations are implementing INT8 AI runtime features. 

We will support INT16 AI runtime features before Cortex Mainnet online, and continuously research on the low bitwidth AI runtime features. Especially, by the time of Cortex Mainnet online, we will support INT8 AI models with INT16 AI runtime features without losing model accuracy, compared to FP32 AI models with FP32 AI runtime features, which means Cortex MainNet can queue cached images to be inferenced up to 2 times as before, without losing model accuracy and computational deterministicity.

#### 3.3 On Supporting Hardwares

A basic implementation will be released on typical ARM chips and FPGA with PCI-E interface before Cortex Mainnet online.

#### 3.4 On Supporting Long-term Evolution of AI models

One of the most important evolution of AI models is the support and optimization of higher dimension tensors.

We will continuously focus on this topic make it compatible with related newly published academic papers.

### References
[Cor(2018)] Coreml document, integrate machine learning models into your app., 2018. URL https://developer.apple.com/documentation/coreml.

[tvm(2018)] Open deep learning compiler stack for cpu, gpu and specialized accelerators: dmlc/tvm, 2018. URL https://github.com/dmlc/tvm.

[Ba et al.(2016)Ba, Kiros, and Hinton] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
1

[Chen et al.(2018)Chen, Moreau, Jiang, Zheng, Yan, Cowan, Shen, Wang, Hu, Ceze, Guestrin, and Krishnamurthy]  Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan,
Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. TVM: An Automated End-to-End
Optimizing Compiler for Deep Learning. arXiv preprint arXiv:1802.04799, February 2018.

[He et al.(2015)He, Zhang, Ren, and Sun] K He, X Zhang, S Ren, and J Sun. Deep residual learning for image recognition. corr, vol. abs/1512.03385, 2015.

[Ioffe & Szegedy(2015)Ioffe and Szegedy] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

[Krizhevsky et al.(2012)Krizhevsky, Sutskever, and Hinton] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information process- ing systems, pp. 1097–1105, 2012.

[Li & Liu(2016)Li and Liu] F Li and B Liu. Ternary weight networks.(2016). arXiv preprint arXiv:1605.04711, 2016.

[NVIDIA(2018)] NVIDIA. Tensorrt document, 2018. URL https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/index.html.

[Sabour et al.(2017)Sabour, Frosst, and Hinton] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. In Advances in Neural Information Processing Systems, pp. 3856–3866, 2017.

[Srivastava et al.(2015)Srivastava, Greff, and Schmidhuber] Rupesh Kumar Srivastava, Klaus Greff, and Ju ̈rgen Schmidhuber. Highway networks. arXiv preprint arXiv:1505.00387, 2015.

[Szegedy et al.(2016)Szegedy, Ioffe, and Vanhoucke] C Szegedy, S Ioffe, and V Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. corr abs/1602.07261. URL http://arxiv.org/abs/1602.07261, 2016.

[Wu & He(2018)Wu and He] Yuxin Wu and Kaiming He. Group normalization. arXiv preprint arXiv:1803.08494, 2018.


