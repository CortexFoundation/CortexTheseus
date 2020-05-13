## Model Performance

Some models have been compiled and the Accuracy and size of params is provided in the following chart for reference.

| Model Name                | Category       | (Top1) Accuracy<br />(original / quantized) |
| ------------------------- | -------------- | ------------------------------------------- |
| ssd_512_mobilenet1.0_coco | detection      | 21.50% / 15.60%                             |
| ssd_512_resnet50_v1_voc   | detection      | 80.27% / 80.01%                             |
| yolo3_darknet53_voc       | detection      | 81.37% / 82.08%                             |
| shufflenet_v1             | classification | 63.48% / 60.45%                             |
| mobilenet1_0              | classification | 70.77% / 66.11%                             |
| mobilenetv2_1.0           | classification | 71.51% / 69.39%                             |

| Model Name                | Params Size | Path                                    |
| ------------------------- | ----------- | --------------------------------------- |
| ssd_512_mobilenet1.0_coco | 23.2M       | /data/mrt/ssd_512_mobilenet1.0_coco_tfm |
| ssd_512_resnet50_v1_voc   | 36.4M       | /data/mrt/ssd_512_resnet50_v1_voc_tfm   |
| yolo3_darknet53_voc       | 59.3M       | /data/mrt/yolo3_darknet53_voc_tfm       |
| shufflenet_v1             | 1.8M        | /data/mrt/shufflenet_v1_tfm             |
| mobilenet1_0              | 4.1M        | /data/mrt/mobilenet1_0_tfm              |
| mobilenetv2_1.0           | 3.4M        | /data/mrt/mobilenetv2_1.0_tfm           |


## Model Preprocess

The data preprocess functions and input shapes are collected with respect to the dataset label in the following chart. for reference.

| Dataset Label | Data Preprocess Function                                     | Input Shape Format                      |
| ------------- | ------------------------------------------------------------ | --------------------------------------- |
| voc           | YOLO3DefaultValTransform(input_size, input_size)             | (batch_size, 3, input_size, input_size) |
| imagenet      | crop_ratio = 0.875<br />resize = $\lceil H/crop\_ratio \rceil$<br />mean_rgb = [123.68, 116.779, 103.939]<br />std_rgb = [58.393, 57.12, 57.375] | (batch_size, 3, input_size, input_size) |
| cifar10       | mean = [0.4914, 0.4822, 0.4465]<br />std = [0.2023, 0.1994, 0.2010] | (batch_size, 3, 32, 32)                 |
| quickdraw     | -                                                            | (batch_size, 1, 28, 28)                 |
| mnist         | mean = 0<br />std = 1                                        | (batch_size, 1, 28, 28)                 |
| trec          | -                                                            | (38, batch_size)                        |
| coco          | SSDDefaultValTransform(input_size, input_size)               | (batch_size, 3, input_size, input_size) |



## Model Output

Some model output introduction is concluded in the following chart.

| Model Type                                                   | Model Output Introduction                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ssd, yolo                                                    | [id, score, bounding_box]<br />bounding_box = (x1, y1, x2, y2) |
| mobilenet, rennet, shufflenet,<br />densenet,alexnet, squeezenet, vgg | score for 1000 classes                                       |
| cifar, quickdraw, mnist                                      | score for 10 classes                                         |
| trec                                                         | score for 6 classes                                          |



Some dataset might need a particular output index to extract the actual value of result which is also enumerated in the following chart.

| Dataset Label | Output Index Converting                                      |
| ------------- | ------------------------------------------------------------ |
| voc           | `map_name, mean_ap = metrics.get()`<br />`acc = {k: v for k,v in zip(map_name, mean_ap)}['mAP']` |
| trec          | `acc = 1. * metrcs["acc"] / metrics["total"]`                |
| coco          | `map_name, mean_ap = metrics.get()`<br />`acc = {k: v for k,v in zip(map_name, mean_ap)}`<br />`acc = float(acc['~~~~ MeanAP @ IoU=[0.50, 0.95] ~~~~\n']) / 100` |

