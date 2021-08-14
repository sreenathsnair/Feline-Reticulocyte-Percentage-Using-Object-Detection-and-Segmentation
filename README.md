# ReticulocytePercentage

This repository contains collaboratory notebooks to perform transfer learning on Feline Reticulocyte dataset. Also it contains the a preprocessing notebook which is used to perform the train, test, val split for Tensorflow 2.0 Object detection models and YOLO.

	Reticulocyte_SegmentationMask.json – Instance segmentation mask for the whole dataset.
	Reticulocyte_Train.json – Instance segmentation mask for the training dataset
	Reticulocyte_Val.json – Instance segmentation mask for the validation dataset.
	Reticulocyte_Test.json – Instance segmentation mask for the test dataset
	Reticulocyte_Preprocessor.ipynb - Code to perform CLAHE, Augmentation. Also, it contains scripts for creating Train, Test and Validation splits and TF records.
	EvaluationUtil.py - Python code to perform post-processing such as duplicate removal and partial object removal
	ReticulocyteCount.ipynb - Notbook to train SSD ResNet 50 with default config
	ReticulocyteCount_SSD_CL_Augmented.ipynb - Notbook to train SSD ResNet 50 with class loss weight
	ReticulocyteCount_SSD_CL_Augmented.ipynb - Notbook to train SSD ResNet 50 with augmented data
	SSDMobileNet_Augmented.ipynb - Notbook to train SSD MobileNet
	ReticulocyteCountEfficientDetD0_CLAug.ipynb - Notbook to train EfficientDetD0
	CenterNetHG104_Augmented.ipynb - Notbook to train CenterNetHG104
	FasterRCNN_Augmented.ipynb - Notebook to train Faster R-CNN
	YOLOv5ReticulocyteAdam.ipynb - Notebook to train YOLO V5 S
	MaskRCNN.ipynb - Notebook to train Mssk RCNN Instance Segmentation Model
	Reticulocyte.py - Python file with customm training and config class for Mask R-CNN
