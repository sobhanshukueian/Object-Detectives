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

| Paper | Link | Type | mAP | Speed | Backbone | Neck | Head | Augmentation | Training Details | Main Idea of Paper | Image | Other Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-CNN | [Paper](https://arxiv.org/abs/1311.2524) | Two-Stage | 58.7 | 5 FPS | VGG-16 | RPN | Fast R-CNN | Horizontal Flipping | SGD | Proposal-based object detection with a Region Proposal Network (RPN) and a Fast R-CNN classifier. | <img src="images/rcnn.png" width="150"> | Notes for R-CNN |
| Fast R-CNN | [Paper](https://arxiv.org/abs/1504.08083) | Two-Stage | 70.0 | 9 FPS | VGG-16 | RPN | Fast R-CNN | Horizontal Flipping, Scale Jittering | SGD | A faster version of R-CNN with a shared convolutional feature map, a Region Proposal Network (RPN), and a Fast R-CNN classifier. | <img src="images/fast_rcnn.png" width="150"> | Notes for Fast R-CNN |
| Faster R-CNN | [Paper](https://arxiv.org/abs/1506.01497) | Two-Stage | 42.1 | 17 FPS | ResNet-50 | RPN | Fast R-CNN | Horizontal Flipping, Scale Jittering, Random Crop | SGD | Improves upon Fast R-CNN by introducing an Region Proposal Network (RPN) that shares the same convolutional features as the object detection network. | <img src="images/faster_rcnn.png" width="150"> | Notes for Faster R-CNN |
| Mask R-CNN | [Paper](https://arxiv.org/abs/1703.06870) | Two-Stage | 64.2 | 4 FPS | ResNet-50 | FPN | Mask R-CNN | Horizontal Flipping, Scale Jittering, Random Crop | SGD | A variant of Faster R-CNN with an additional branch for predicting object masks in parallel with the existing branch for bounding box recognition. | <img src="images/mask_rcnn.png" width="150"> | Notes for Mask R-CNN |



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

