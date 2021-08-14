# ReticulocytePercentage

This repository contains collaboratory notebooks to perform transfer learning on Feline Reticulocyte dataset. Also it contains the a preprocessing notebook which is used to perform the train, test, val split for Tensorflow 2.0 Object detection models and YOLO.

1.	Reticulocyte_SegmentationMask.json – Instance segmentation mask for the whole dataset.
2.	Reticulocyte_Train.json – Instance segmentation mask for the training dataset
3.	Reticulocyte_Val.json – Instance segmentation mask for the validation dataset.
4.	Reticulocyte_Test.json – Instance segmentation mask for the test dataset
5.	Reticulocyte_Preprocessor.ipynb - Code to perform CLAHE, Augmentation. Also, it contains scripts for creating Train, Test and Validation splits and TF records.
6.  EvaluationUtil.py - Python code to perform post-processing such as duplicate removal and partial object removal
7.	ReticulocyteCount.ipynb - Notbook to train SSD ResNet 50 with default config
8.	ReticulocyteCount_SSD_CL_Augmented.ipynb - Notbook to train SSD ResNet 50 with class loss weight
9.	ReticulocyteCount_SSD_CL_Augmented.ipynb - Notbook to train SSD ResNet 50 with augmented data
10.	SSDMobileNet_Augmented.ipynb - Notbook to train SSD MobileNet
11.	ReticulocyteCountEfficientDetD0_CLAug.ipynb - Notbook to train EfficientDetD0
12.	CenterNetHG104_Augmented.ipynb - Notbook to train CenterNetHG104
13.	FasterRCNN_Augmented.ipynb - Notebook to train Faster R-CNN
14.	YOLOv5ReticulocyteAdam.ipynb - Notebook to train YOLO V5 S
15.	MaskRCNN.ipynb - Notebook to train Mssk RCNN Instance Segmentation Model
16.	Reticulocyte.py - Python file with customm training and config class for Mask R-CNN
