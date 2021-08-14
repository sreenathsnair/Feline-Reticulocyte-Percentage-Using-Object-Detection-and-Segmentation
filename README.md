# ReticulocytePercentage

This repository contains collaboratory notebooks to perform transfer learning on Feline Reticulocyte dataset. Also it contains the a preprocessing notebook which is used to perform the train, test, val split for Tensorflow 2.0 Object detection models and YOLO.

1. Reticulocyte_Preprocessor.ipynb - Code to perform CLAHE, Augmentation. Also it contains scripts for creating Train, Test and Validation splits and TF records.
2. EvaluationUtil.py - Python code to perform post-procesing such as duplicate removal and partial object removal
3. ReticulocyteCount.ipynb - Notbook to train SSD ResNet 50 with default config
4. ReticulocyteCount_SSD_CL_Augmented.ipynb - Notbook to train SSD ResNet 50 with class loss weight
5. ReticulocyteCount_SSD_CL_Augmented.ipynb - Notbook to train SSD ResNet 50 with augmented data
6. SSDMobileNet_Augmented.ipynb - Notbook to train SSD MobileNet
7. ReticulocyteCountEfficientDetD0_CLAug.ipynb - Notbook to train EfficientDetD0
8. CenterNetHG104_Augmented.ipynb - Notbook to train CenterNetHG104
9. FasterRCNN_Augmented.ipynb - Notebook to train Faster R-CNN
10. YOLOv5ReticulocyteAdam.ipynb - Notebook to train YOLO V5 S
11. MaskRCNN.ipynb - Notebook to train Mssk RCNN Instance Segmentation Model
12. Reticulocyte.py - Python file with customm training and config class for Mask R-CNN
