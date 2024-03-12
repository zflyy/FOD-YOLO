# README
## An Enhanced and Lightweight Small-scale Foreign Object Debris Detection Model based on YOLOv8

[![Overall Structure of the FOD-YOLO](https://github.com/Dafei-Zhang/FOD-YOLO/blob/main/imgs/overall%20structure.jpg)](https://github.com/Dafei-Zhang/FOD-YOLO/blob/main/imgs/overall%20structure.jpg)

## Abstract
Object detection has been extensively applied in various fields due to the efficient performance of convolutional neural networks (CNNs). However, for certain special fields, such as foreign boject debris (FOD) detection, direct utilization of generic detectors, still faced challenges including false positives, false negatives, and lack of lightweight model design although capable of achieving certain results. In this article, we propose a FOD detection model named as FOD-YOLO, which can improve the detection accuracy of small FOD items and simultaneously decrease the parameters of the implemented model. The proposed FOD-YOLO follows the overall framework of YOLOv8 and can be viewed as one of its improved variants. Firstly, to compensate for the loss of information regarding small objects during feature extraction, high-resolution feature maps were incorporated into the detection layer to fuse multiscale features and the large object detection layer was removed from the model. Secondly, a Lightweight-Backbone with strong feature extraction ability was developed by introducing Lightweight Downsampling Convolution (LDConv) modules, Deformable Convolution v3 to reconstruct C2f (DCNv3_C2f) modules, and a Bi-level Routing Attention (BRA) mechanism. Subsequently, the lightweight Slim-Head was proposed by introducing slim-neck and Group-RepConv with Efficient Channel Attention Mechanism Head (GREHead) modules. Ultimately, the Complete Intersection over Union (CIoU) loss function was replaced with the Minimum Point Distance Intersection over Union (MPDIoU) loss function, aiming to accelerate Bounding Box Regression (BBR) convergence speed and enhance regression accuracy. The experimental results demonstrate that the proposed FOD-YOLO can achieve better mean Average Precision (mAP), especially in detecting small FOD items, over the other state-of-the-art methods with small parameters.

## Innovations
- Enhanced Multiscale Feature Fusion
- Lightweight-Backbone
- Slim-Head
- MPDIoU Loss Function

## Environment Setup
- Operating System: Ubuntu 18.04
- CPU: Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
- GPU: NVIDIA RTX 3090Ti with 24GB memory
- Programming Language: Python 3.9.12
- IDE: PyCharm
- Deep Learning Framework: PyTorch 2.0.0
- GPU Support: CUDA 11.7
- Virtual Environment: Anaconda 4.13.0
- Framework Version: Ultralytics 8.0.114

## Training Parameters
- Epochs: 600
- Batch Size: 32
- Initial Learning Rate: 1e-2
- Momentum: 0.937
- Weight Decay: 1e-4
- Number of Threads: 12
- Optimizer: SGD

## Description
This repository contains code for conducting deep learning experiments using the specified environment and parameters. The setup aims to balance model learning and computational resource consumption effectively. The parameters have been carefully chosen to control the learning process, enhance training stability, and prevent overfitting.

## Getting Started
```bash
# Compile DCNv3
cd ultralytics-main/ultralytics/nn/modules/ops_dcnv3
sh ./make.sh

# Change directory, install ultralytics
cd ultralytics-main
pip install -e .

# Install the required dependencies listed in `requirements.txt`.

# Set up the environment based on the provided specifications.

# Run the training script using the specified parameters.
```

## Prepare Dataset
- **FOD-Tiny Dataset**
  
  **Baidu Disk**: [https://pan.baidu.com/s/1CQPwVK-CY8kcgoD-QyHz4w](https://pan.baidu.com/s/1CQPwVK-CY8kcgoD-QyHz4w)
  
  category: Plastic Pipe, Plug, Aluminium Alloy Fitting, Plastic Buckle, Motor Aluminum Tube, Elliptical Iron Pipe, Circular Steel Column, Weight, Iron Ball, Golf, Hexagon Nut, Ball Nut
  
  A detailed introduction to the FOD-Tiny data set:
  
  | Split   | Total | Train | Val  | Test |
  |---------|:-----:|:-----:|:----:|:----:|
  | number  | 5748  | 4137  | 1035 | 576  |

## Train and Test
```python
# Train
yolo task=detect mode=train model=FOD-YOLO.yaml data=data.yaml epochs=600
# Test
yolo predict model=best.pt source='imgs/fod.jpg'

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```
See YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python) for more examples.

## Results

Test on GTX 3090Ti GPU:

| Model     | Backbone     | mAP@0.5:0.95 | Parameters/M | FPS |
|-----------|:------------:|:------------:|:------------:|:---:|
| YOLOv5n   | CSPDarknet53 | 64.3         | 2.51         | 128 |
| YOLOv6n   | EfficientRep | 60.5         | 4.24         | 63  |
| YOLOv7-tiny| ELAN        | 60.3         | 6.05         | 86  |
| YOLOv8n   | DarkNet-53   | 64.5         | 3.01         | 196 |
| **FOD-YOLO**| -          | 68.7         | 1.60         | 78  |

## Acknowledgements
- Special thanks to the developers of PyTorch, CUDA, Anaconda, and Ultralytics for their contributions to the deep learning ecosystem.
