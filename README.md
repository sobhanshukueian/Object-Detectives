<p align="center">
  <img src="https://img.shields.io/badge/Object%20Detection-Papers%20and%20Experiments-blueviolet.svg" alt="Object Detection Papers and Experiments">
</p>

<h1 align="center">Object Detection [👁️] Papers and Experiments</h1>

<p align="center">
  A survey of object detection papers with notes, links, and other details, along with run experiments and results.
</p>

<p align="center">
  <a href="#books-papers">Papers</a> •
  <a href="#mag_right-comparison-between-yolov4-and-yolov5">YOLO-V4 VS YOLO-V5</a> •
  <a href="#computer-applications">Applications</a> •
  <a href="#chart_with_upwards_trend-run-experiments-and-results">Experiments</a>
</p>



## :books: Papers


<h3><b>R-CNN</b></h3>

- **Paper:** [R-CNN Paper](https://arxiv.org/abs/1311.2524)
- **Type:** Two-Stage
- **mAP:** 58.7
- **Speed:** 5 FPS
- **Backbone:** VGG-16
- **Neck:** RPN
- **Head:** Fast R-CNN
- **Main Idea:** Proposal-based object detection with a Region Proposal Network (RPN) and a Fast R-CNN classifier.

<h3><b>Fast R-CNN</b></h3>

- **Paper:** [Fast R-CNN Paper](https://arxiv.org/abs/1504.08083)
- **Type:** Two-Stage
- **mAP:** 70.0
- **Speed:** 9 FPS
- **Backbone:** VGG-16
- **Neck:** RPN
- **Head:** Fast R-CNN
- **Augmentation:** Horizontal Flipping, Scale Jittering
- **Main Idea:** A faster version of R-CNN with a shared convolutional feature map, a Region Proposal Network (RPN), and a Fast R-CNN classifier.


<h3><b>You Only Look Once: Unified, Real-Time Object Detection</b></h3>

- **Paper:** [YOLO v1](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
- **Publish date:** 2016
- **Type:** One Stage
- **mAP:** 63.40%
- **Speed:** 45 fps
- **Backbone:** "designed their own convolutional backbone which was inspired by GoogLeNet
- **Head:** 2 fully connected layer with grid cell and bounding boxes
- **Augmentation:** For data augmentation introduce random scaling and translations of up to 20% of the original image size. also randomly adjust the exposure and saturation of the image by up to a factor of 1:5 in the HSV color space
- **Notes:** Frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance. YOLO sees the entire image during training and test time so it implicitly encodes contextual  information about classes as well as their appearance divides the input image into an S × S grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object. Each grid cell predicts B bounding boxes and confidence scores for those boxes"
- ![image](https://github.com/sobhanshukueian/Object-Detectives/assets/47561760/fb852895-1e27-4e77-ac0f-f36b827f03ae)
  
<h3><b>SSD: Single Shot MultiBox Detector</b></h3>

- **Paper:** [SSD Paper](https://arxiv.org/pdf/1512.02325)
- **Publish date:** 2019
- **Type:** One Stage
- **mAP:** "300*300 input 74.3% mAP on VOC2007 and 500*500 input 76.9%"
- **Speed:** "input 300*300 on VOC2007 59 FPS on nvidia titan X
- **Backbone:** VGG-16
- **Augmentations:** Use the entire original input image. Sample a patch so that the minimum jaccard  overlap with the objects is 0.1, 0.3, 0.5, 0.7, or 0.9. Randomly sample a patch.
- ![image](https://github.com/sobhanshukueian/Object-Detectives/assets/47561760/337efba5-35c6-4f66-9b8e-c27f5a40cda6)

  
<h3><b>YOLO9000: Better, Faster, Stronger</b></h3>

- **Paper:** [YOLO v2 Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)
- **Publish date:** 2017
- **Type:** One Stage
- **mAP:** 78.6 on voc2007
- **Speed:** 40 FPS
- **Backbone:** Darknet-19
- **Augmentation:** Use a similar data augmentation to YOLO and SSD with random crops, color shifting, etc.
- **Training Details:** Train the network for 160 epochs with a starting learning rate of 10e−3, dividing it by 10 at 60 and 90 epochs. use a weight decay of 0.0005 and momentum of 0.9.
- **Notes:** "1) adding batch normalization on all of the  convolutional layers in YOLO get more than 2% improvement in mAP.
2) first fine tune the classification network at the full 448 × 448 resolution for 10 epochs on ImageNet. then fine tune the resulting network on detection. This gives us an increase of almost 4% mAP.
3) Convolutional With Anchor Boxes: remove the fully connected
 layers from YOLO and use anchor boxes to predict bounding boxes.Even though the mAP decreases, the increase in recall happens.
4) run k-means clustering on the training set bounding boxes to automatically find good priors.
5) Instead of predicting offsets follow the approach of YOLO and  predict location coordinates relative to the locationof the grid cell. This bounds the ground truth to fall between 0 and 1. We use a logistic.
5) Passthrough layer that brings features from an earlier layer at 26 × 26 resolution.
6) Training Instead of fixing the input image size change the network every few iterations. Every 10 batches our network randomly chooses a new image dimension size.
7) Combine datasets using WordTree can train our joint model on classification and detection.

<h3><b>YOLOv3: An Incremental Improvement</b></h3>

- **Paper:** [YOLO v3 Paper](https://arxiv.org/pdf/1804.02767)
- **Publish date:** 2018
- **Type:** One Stage
- **mAP:** 51.5%
- **Speed:** 78 fps
- **Backbone:** Darknet-53
- **Training Details:** Train on full images with no hard negative mining or any of that stuff. We use multi-scale training, lots of data augmentation, batch normalization, all the standard stuff. use the Darknet neural network framework for training and testing
- **Notes:** Better at detecting smaller objects and stronger than previous versions!.for detect smaller objects:YOLO v3, in total uses 9 anchor boxes. Three for each scale. If you’re training YOLO on your own dataset, you should go about using K-Means clustering to generate 9 anchors.

<h3><b>YOLOv4: Optimal Speed and Accuracy of Object Detection</b></h3>

- **Paper:** [YOLO v4 Paper](https://arxiv.org/pdf/2004.10934)
- **Publish date:** 2020
- **Type:** One Stage
- **mAP:** 43.5% AP (65.7% AP50)
- **Speed:** 65 FPS on Tesla V100
- **Backbone:** CSPDarknet53
- **Head:** YOLO-v3
- **Notes:** YoloV4 is an important improvement of YoloV3, the implementation of a new architecture in the Backbone and the modifications in the Neck have improved the mAP(mean Average Precision) by 10% and the number of FPS(Frame per Second) by 12%. In addition, it has become easier to train this neural network on a single GPU.
  

<h3><b>YOLOX: Exceeding YOLO Series in 2021</b></h3>

- **Paper:** [YOLO X Paper](https://arxiv.org/pdf/2107.08430.pdf)
- **Publish date:** 2021
- **Type:** One Stage
- **Backbone:** DarkNet53
- **Neck:** SPP
- **Head:** Anchor free
- **Augmentations:** Random Horizontal Flip, Color Jitter, discard the RandomResizedCrop strategy, because we found the RandomResizedCrop is kind of overlapped with the planned mosaic augmentation.
- **Training Details:** EMA weights updating, cosine lr schedule, IoU loss and IoU-aware branch.300 epochs with 5 epochs warmup on COCO train2017" with
- **Notes:**  (1). Replacing YOLO’s head with a decoupled one greatly improves the converging speed . 2). The decoupled head is essential to the end-to-end version of YOLO)
- ![image](https://github.com/sobhanshukueian/Object-Detectives/assets/47561760/af7c0cc3-1eb8-45bf-b158-6a8e8b7b5a22)

<h3><b>Densely Connected Convolutional Networks</b></h3>

- **Paper:** [DenseNet Paper](https://arxiv.org/pdf/1608.06993.pdf)
- **Publish date:** 2018
- **Main Idea:** The core of DenseNet is using Dense blocks which is an essential of the idea behind it all. The core idea is that within a block, it contains multiple layers. All previous attempts before this paper only used the layers in sequential manner. An output of a layer is fed to the next layer.
- **Advantages:** they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.One big advantage of DenseNets is their improved flow of information and gra dients throughout the network,Each layer has direct access to the gradients from the loss function and the original input signal,eading to an implicit deep supervision.Further, we also observe that dense connections have a regularizing effect.
- **Notes:** we never combine features through summation before they are passed into a layer; in stead, we combine features by concatenating them.Further, we also observe that dense connections have a regularizing effect,One explanation for the improved accuracy of dense convolutional networks may be that individual layers receive additional supervision from the loss function through the shorter connections.
- ![image](https://github.com/sobhanshukueian/Object-Detectives/assets/47561760/41abc2b8-3f3c-4dfe-bc0e-87ea9b4572ef)

<h3><b>CSPNET: A NEW BACKBONE THAT CAN ENHANCE LEARNING CAPABILITY OF CNN</b></h3>

- **Paper:** [CSPNET Paper](https://arxiv.org/pdf/1911.11929.pdf)
- **Publish date:** 2019
- **Main Idea:** The main purpose of designing CSPNet is to enable this architecture to achieve a richer gradient combination while reducing the amount of computation. This aim is achieved by partitioning feature map of the base layer into two parts and then merging them through a proposed cross-stage hierarchy. Our main concept is to make the gradient flow propagate through different network paths by splitting the gradient flow. In this by splitting the gradient flow
- **Advantages:** 1) Strengthening learning ability of a CNN. 2) Removing computational bottlenecks. 3) Reducing memory costs
- ![image](https://github.com/sobhanshukueian/Object-Detectives/assets/47561760/b32ed182-c45f-4ff2-88c2-7ea6047b82cc)

<h3><b>Path Aggregation Network for Instance Segmentation</b></h3>

- **Paper:** [PA Net Paper](https://arxiv.org/pdf/1803.01534.pdf)
- **Publish date:** 2018
- **Main Idea:** Specifically, we enhance the entire feature hierarchy with accurate localization signals in lower layers by bottom-up path augmentation,
- **Advantages:** Features in multiple levels together are helpful for accurate prediction.
- **Notes:** We use max operation to fuse features from different levels, which letsvnetwork select element-wise useful information.
- ![image](https://github.com/sobhanshukueian/Object-Detectives/assets/47561760/f56cf01b-48ba-4945-b2ab-ee4e053d9d83)

<h3><b>EfficientDet: Scalable and Efficient Object Detection</b></h3>

- **Paper:** [EfficientDet Paper](https://arxiv.org/pdf/1911.09070.pdf)
- **Publish date:** 2020
- **Main Idea:** They propose a weighted bi-directional feature pyra mid network (BiFPN), which allows easy and fast multi scale feature fusion; Second, They propose a compound scal ing method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class pre diction networks at the same time.
- **Advantages:** Aiming at optimizing both accuracy and efficiency, we would like to develop a family of models that can meet a wide spectrum of resource constraints.
- **Notes:** While fusing different input features, most previous works simply sum them up without distinction;which introduces learnable weights to learn the importance of different input features, while repeatedly applying top down and bottom-up multi-scale feature fusion.
- ![image](https://github.com/sobhanshukueian/Object-Detectives/assets/47561760/87a27600-2efe-4838-b64a-93be952b5dcc)

<h3><b>Squeeze-and-Excitation Networks</b></h3>

- **Paper:** [SE Net Paper](https://arxiv.org/pdf/1709.01507.pdf)
- **Publish date:** 2019
- **Main Idea:** They focus instead on the channel relationship and propose a novel architectural unit, which they term the “Squeeze-and-Excitation” (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
- **Advantages:** They show that these blocks can be stacked together to form SENet architectures that generalise extremely effectively across different datasets. They further demonstrate that SE blocks bring significant improvements in performance for existing state-of-the-art CNNs at slight additional computational cost.
- ![image](https://github.com/sobhanshukueian/Object-Detectives/assets/47561760/f8f60818-4e8f-4086-9b87-4c8c5c673939)


## :mag_right: Comparison between YOLOv4 and YOLOv5

### YOLO-V4

![image](https://user-images.githubusercontent.com/47561760/233497319-e7bd662b-09f2-4b88-8df8-b60ef99cf73c.png)

- **Backbone**: `CSP-Darknet53`
- **Neck**: `SPP`, `PAN`
- **Head**: `YOLOv3 Head`

#### What they checked :

- **Input**: `Image, Patches, Image Pyramid`
- **Backbones**: `VGG16 , ResNet-50, SpineNet, EfficientNet-B0/B7, CSPResNeXt50, CSPDarknet53`
- **Neck**:
    - Additional blocks: `SPP, ASPP, RFB, SAM`
    - Path-aggregation blocks: `FPN, PAN, NAS-FPN, Fully-connected FPN, BiFPN, ASFF, SFAM`
- **Heads**:
    - Dense Prediction (one-stage): `RPN, SSD, YOLO, RetinaNet (anchor-based) CornerNet, CenterNet, MatrixNet, FCOS(anchor free)`
    - Sparse Prediction (two-stage): `Faster R-CNN, R-FCN, Mask RCNN (anchor-based), RepPoints (anchor free)`

### YOLO-V4 USE:

- **Bag of Freebies (BoF) for backbone**: `CutMix and Mosaic data augmentation, DropBlock regularization, Class label smoothing`
- **Bag of Specials (BoS) for backbone**:`Mish activation, Cross-stage partial connections (CSP), Multi-input weighted residual connections (MiWRC)`
- **Bag of Freebies (BoF) for detector**: `CIoU-loss, CmBN, DropBlock regularization, Mosaic data augmentation, Self-Adversarial Training, Eliminate grid sensitivity, Using multiple anchors for single ground truth, Cosine annealing scheduler, Optimal hyperparameters, Random training shapes`
- **Bag of Specials (BoS) for detector:**  `Mish activation, SPP-block, SAM-block, PAN path-aggregation block, DIoU-NMS`\

---

- **Drop Block** :
    
    [https://arxiv.org/abs/1810.12890](https://arxiv.org/abs/1810.12890)
    
    ![image](https://user-images.githubusercontent.com/47561760/233497782-08cab4b9-64da-46bb-884c-3aec67ff7260.png)
    
    ![image](https://user-images.githubusercontent.com/47561760/233497795-c1085cef-216b-469f-bab6-4caa7db8ef4d.png)
    
- **Mosaic and MixUp Augmentations** :
    
    ![image](https://user-images.githubusercontent.com/47561760/233497888-fa9532f2-e8e4-4d40-90b1-fc7f293dab0d.png)
    
- ****CBAM: Convolutional Block Attention Module :****
    
    [https://arxiv.org/abs/1807.06521](https://arxiv.org/abs/1807.06521)
    
    ![image](https://user-images.githubusercontent.com/47561760/233497947-5a56b2b0-9c59-4799-8457-285f5526e358.png)

---

### YOLO-V5

- **Backbone**: `New CSP-Darknet53`
- **Neck**: `SPPF`, `New CSP-PAN`
- **Head**: `YOLOv3 Head`

[https://user-images.githubusercontent.com/31005897/158507974-2c275082-95bd-4c2c-839d-6aec19121f02.png](https://user-images.githubusercontent.com/31005897/158507974-2c275082-95bd-4c2c-839d-6aec19121f02.png)

---

### YOLO-v4 vs YOLO-v5

Yolov4: `Mish`  ||   Yolov5: `SiLU`

![image](https://user-images.githubusercontent.com/47561760/233498001-6a85f628-5c19-4875-883c-1dd09c2a2425.png)
![image](https://user-images.githubusercontent.com/47561760/233498032-05537ce4-78bd-464a-b74a-61e2c9686d87.png)

Yolov4: `SPP` || Yolov5: `SPPF`

![image](https://user-images.githubusercontent.com/47561760/233498066-3f6caf0f-f67a-4598-8b90-ce2d9d1240ce.png)

---

#

## :computer: Applications

Object detection has a wide range of applications, including:

- Retail
- Self-driving cars
- Medical diagnosis
- Robotics
- Animal detection in Agriculture 
- Optical character recognition
- Automated CCTV


## :chart_with_upwards_trend: Run Experiments and Results

| Paper | GPU (Colab) | CPU (Colab) | Mobile (CPU) | Mobile (GPU) | Laptop (CPU) |
| --- | --- | --- | --- | --- | --- |
| | FPS - mAP | FPS - mAP | FPS - mAP | FPS - mAP | FPS - mAP |
| YOLO v4	| 9fps - 65 | - | - | - | - |
| YOLO v3 tiny | - | - | - | - | 14fps - 33.1 |
| YOLO v5 medium |10fps -	67|2fps - 67 | - | - |4fps - 67|
| YOLO v5 small	| 33fps -	62 | 3fps -	62 | - | - |9fps -	62 |
| YOLO v5 nano	 |58fps -	46 | 11fps -	46 | - | - | 20fps -	46 |
| YOLOX tiny	| - | - | 10fps -	32.8 |7fps - 32.8| -  |
| YOLOX nano	| - | - | 20 fps -	25.8 |	13 fps -	25.8 | - |
| YOLO v5 320-lite-e	| -| - | 27 fps -	35.1 |	17 fps -	35.1 | 20fps -	33.7 |
| YOLO v5 416-lite-e	|- | -| 20 fps -	35.1 |	12 fps -	35.1 | - |
| YOLO v5 320-lite-i8e	|- |- | 21 fps -	35.1 |	22 fps -	35.1 |   -|
| YOLO v5 416-lite-i8e	| -| -| 17 fps -	35.1 |	17 fps -	35.1 | - |
| YOLO v5 416-lite-s	| - |  -| 18 fps -	42 |	11 fps -	42 | - |
| YOLO v5 416-lite-i8s	| - | - | 19 fps -	42 |	17 fps -	42 | -|
| YOLO v5 512-lite-c	 | - | - | 9 fps -	50.9 |	6 fps -	50.9 | 12fps -	44 |

## 📝 Contributing

We welcome contributions to this repository! Please open an issue or pull request for any suggestions or changes.


# ⭐️ Please Star This Repo ⭐️

If you found this project useful or interesting, please consider giving it a star on GitHub! This helps other users discover the project and provides valuable feedback to the maintainers.

Thank you for your support!
