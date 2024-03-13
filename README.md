# README
## An Improved and Lightweight Small-scale Foreign Object Debris Detection Model based on YOLOv8

![Overall Structure of the FOD-YOLO](https://github.com/Dafei-Zhang/FOD-YOLO/blob/main/ultralytics-main/imgs/overall%20structure.jpg)

## Abstract
Foreign object debris (FOD) on runways may cause irreparable damage and detecting FOD in intellectual technology has attracted more and more attention. Due to the limitation of the equipment, the main goal of the FOD detection methods is to obtain as high accuracy as possible by employing the model with as few parameters as possible. In this article, we propose a FOD detection model named FOD-YOLO, which can improve the detection accuracy of small FOD items and simultaneously decrease the parameters of the implemented model. The proposed FOD-YOLO follows the overall framework of YOLOv8 and can be viewed as one of its improved variants. Firstly, to compensate for the loss of information regarding small objects during feature extraction, high-resolution feature maps were incorporated into the detection layer to fuse multiscale features, and the large object detection layer was removed from the model. Secondly, a Lightweight-Backbone with strong feature extraction ability was developed by introducing Lightweight Downsampling Convolution (LDConv) modules, Deformable Convolution v3 to reconstruct C2f (DCNv3_C2f) modules, and a Bi-level Routing Attention (BRA) mechanism. Subsequently, the lightweight Slim-Head was proposed by introducing slim-neck and Group-RepConv with Efficient Channel Attention Mechanism Head (GREHead) modules. Ultimately, To validate the effectiveness of the proposed algorithm, comparative experiments were conducted using the self-constructed small target FOD dataset. The results demonstrate that the proposed FOD-YOLO can achieve better accuracy, over the other state-of-the-art methods with small parameters.

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
  A detailed introduction to split the FOD-Tiny dataset:
  
  **Category Information**
  
  | No. | Category               | Number |
  |-----|------------------------|--------|
  | 1   | Plastic Pipe           | 385    | 
  | 2   | Plug                   | 425    | 
  | 3   | Aluminium Alloy Fitting| 477    | 
  | 4   | Plastic Buckle         | 763    | 
  | 5   | Motor Aluminum Tube    | 562    | 
  | 6   | Elliptical Iron Pipe   | 542    | 
  | 7   | Circular Steel Column  | 429    | 
  | 8   | Weight                 | 502    | 
  | 9   | Iron Ball              | 361    | 
  | 10  | Golf                   | 335    | 
  | 11  | Hexagon Nut            | 383    | 
  | 12  | Ball Nut               | 584    |

**Samples Distribution of FOD-Tiny Data:**
 
The sizing range of targets in this study was defined based on absolute size dimensions. Within the MS COCO (Microsoft Common Objects in Context) dataset [11], targets were classified into three categories: small targets, medium targets, and large targets. Small targets referred to objects with dimensions smaller than 32 × 32 pixels, medium targets referred to objects with dimensions ranging from 32×32 to 96 × 96 pixels, and large targets referred to objects with dimensions larger than 96 × 96 pixels. The data statistics presented in the follow Table revealed that the dataset contained the highest number of small-sized targets, with a total of 5134 images containing small targets. Following that, 537 images included medium-sized targets, and finally, the dataset comprised 77 images featuring large-sized targets.

  | Type   | Area           | Number |
  |--------|:--------------:|:------:|
  | Small  | 0 < a <= 32*32 | 5134   |
  | Medium | 32*32 < a <= 96*96 | 537  |
  | Large  | a > 96*96      | 77     |


**Split the FOD-Tiny Data:**
  
  | Split   | Total | Train | Val  | Test |
  |---------|:-----:|:-----:|:----:|:----:|
  | Number  | 5748  | 4137  | 1035 | 576  |

  **Related Dataset Link**: 
### Related Datasets

  - **FOD-Tiny Dataset**
    - **Dataset Link**: [FOD-Tiny Dataset Link](https://pan.baidu.com/s/1CQPwVK-CY8kcgoD-QyHz4w)

  - **FOD-A Dataset**
    - **Dataset Link**: [FOD-A Dataset Link](https://github.com/FOD-UNOmaha/FOD-data)

  - **AI-TOD Dataset**
    - **Dataset Link**: [AI-TOD Dataset Link](https://github.com/jwwangchn/AI-TOD)

  - **VisDrone2019 Data**
    - **Dataset Link**: [VisDrone2019 Data Link](https://github.com/VisDrone/VisDrone-Dataset)
    
## Train and Test
```python
# Train
yolo task=detect mode=train model=FOD-YOLO.yaml data=data.yaml epochs=600
# Test
yolo predict model=best.pt source='imgs/fod.jpg'

from ultralytics import YOLO

# Load a model
model = YOLO("FOD-YOLO.yaml")  # build a new model from scratch
model = YOLO("best.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="Fod_Tiny.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("imgs/fod.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```
See YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python) for more examples.

## Results

Test on Tiny_Fod Dataset on GTX 3090Ti GPU:

| Model     | Backbone     | mAP@0.5:0.95 | Parameters/M | FPS |
|-----------|:------------:|:------------:|:------------:|:---:|
| YOLOv5n   | CSPDarknet53 | 64.3         | 2.51         | 128 |
| YOLOv6n   | EfficientRep | 60.5         | 4.24         | 63  |
| YOLOv7-tiny| ELAN        | 60.3         | 6.05         | 86  |
| YOLOv8n   | DarkNet-53   | 64.5         | 3.01         | 196 |
| **FOD-YOLO**| -          | 68.7         | 1.60         | 78  |

## Acknowledgements
- Special thanks to the developers of PyTorch, CUDA, Anaconda, and Ultralytics for their contributions to the deep learning ecosystem.


## Main References
1. [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)
2. [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)
3. [FOD-UNOmaha FOD Data GitHub Repository](https://github.com/FOD-UNOmaha/FOD-data)
4. [AI-TOD GitHub Repository](https://github.com/jwwangchn/AI-TOD)
5. [VisDrone Dataset GitHub Repository](https://github.com/VisDrone/VisDrone-Dataset)
6. Li, C., Li, L., Jiang, H., Weng, K., Geng, Y., Li, L., Ke, Z., Li, Q., Cheng, M., Nie, W., Li, Y.: *YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications*. arXiv preprint, (2022). [Paper](https://doi.org/10.48550/arXiv.2209.02976)
7. Wang, C. Y., Bochkovskiy, A., Liao, H. Y.: *YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors*. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7464-7475 (2023). [Paper](https://doi.org/10.1109/CVPR52729.2023.00721)
