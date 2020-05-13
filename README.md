# cvm-runtime
CVM Runtime



## Latency

model|  Jetson Nano \- Cortex\-A57(s) | Intel E5\-2650(s) |  Jetson Nano \- GPU(128 CUDA Cores)(s) | 1080Ti(3584 CUDA Cores)(s)
-|-|-|-|-
yolo_tfm | | | 1.076 | 0.043
resnet50_mxg | 1.2076| 0.3807| 0.147 | 0.009
resnet18_v1 | |  | 0.055 | 0.004
qd10_resnet20_v2 ||  | 0.064 | 0.010
resnet50_v2 |1.4674| 0.5005 | 0.185 | 0.010
qd10_resnet20_v2|0.2944|0.1605 | 0.065 | 0.012
trec | 0.0075| 0.0028 | 0.002 | 0.001
dcnet_mnist_v1|0.0062|0.0057 | 0.002 | 0.001
mobilenetv1.0_imagenet|0.3508| 0.1483| 0.039  | 0.002
resnet50_v1_imagenet|1.2453| 0.3429 | 0.150 | 0.009
animal10 | 0.3055 | 0.1466 | 0.065 | 0.010
vgg16_gcv|4.3787| 0.6092 | 0.713 | 0.021
sentiment_trec|0.0047| 0.0022 |  0.002 | 0.001
vgg19_gcv|5.1753| 0.7513 | 0.788 | 0.023
squeezenet_gcv1.1|0.3889|  0.0895 |  0.044 | 0.002
squeezenet_gcv1.0|0.1987| 0.1319 | 0.064 | 0.003
shufflenet|1.4575| 0.7697 | 0.140 | 0.004
ssd| | |0.773 | 0.030
ssd_512_mobilenet1.0_coco_tfm| | | 0.311 | 0.016
ssd_512_mobilenet1.0_voc_tfm| | | 0.220 | 0.014
