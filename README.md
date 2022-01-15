# YOLOV1-PyTorch-Implementation-from-Scratch

This repository implements YOLOv1 from scratch using geometry dataset (https://drive.google.com/drive/u/1/folders/1XXIabiQa0t9gvOMBOWcX75zZuLRMxQvv) with 3 geometry types : triangle, rectangle, circle. The task is to find the bounding boxes for shapes and classify them. The paper of YOLOv1 could be found here : https://arxiv.org/abs/1506.02640<br>
![a](https://github.com/pbcquoc/yolo/blob/master/image/dataset.png)

### 1. Backbones
The model uses vgg16 as its backbone (https://arxiv.org/abs/1409.1556)

### 2. Yolov1 configs
- grid size : 7
- boxes per grid (to predict) : 2
- input size : 224
- num classes : 3 (0: triangle, 1: rectangle, 2: circle)

### 3. Training parameters
- batchsize : 64
- learning rate : 0.001
- epochs : 300
- learning rate scheduler rate : 0.5
- learning rate scheduler step : 60
- weight decay : 5e-4
- optimizer : Adam

### 4. Training notebooks : https://colab.research.google.com/drive/1uG2YpYUWdpLFh2K9E3KoGQvSm4-A8TR3?authuser=1#scrollTo=6Jc7CjTFa1gU

### 5. Training results : 
- Best val iou : 0.82
- training histories : https://github.com/Uchiha-Hieu/YOLOV1-PyTorch-Implementation-from-Scratch/tree/main/training_hist
- Losses : <br>
![](https://github.com/Uchiha-Hieu/YOLOV1-PyTorch-Implementation-from-Scratch/blob/main/Loss.png)

- IOUs : <br>
![](https://github.com/Uchiha-Hieu/YOLOV1-PyTorch-Implementation-from-Scratch/blob/main/Iou.png)
