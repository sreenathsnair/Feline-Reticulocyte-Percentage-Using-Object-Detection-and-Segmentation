# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:01:47 2021

@author: Sreenath
"""
import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET
import cv2 as cv

# Function Name: retrieve_bounding_box_details
# Input        : path to dataset directory.
# Output       : Return a bounding box dataframe which has the ground truth
#                bounding box details for each objects in the datset. Also
#                return the list of image file names and label file names
# Description  : This method will parse each label file and extract the 
#                ground truth data.
def retrieve_bounding_box_details(data_directory):
    label_data = []
    _, _, allfiles = next(os.walk(data_directory))
    image_files_list = [fname for fname in allfiles if '.jpg' in fname]
    label_files_list = [fname for fname in allfiles if '.xml' in fname]
    for label in label_files_list:
        xml_tree = ET.parse(data_directory+label)
        root = xml_tree.getroot()
        image_file = root.find('filename').text
        size_elem =root.find('size')
        height = float(size_elem.find('height').text)
        width =  float(size_elem.find('width').text)

        # For every bounding box get the details (x1, y1, x2, y2)
        for bbox in root.findall('object'):
            # Bounding box coordinates, here the coordinate are scaled
            bounding_box = bbox.find('bndbox')
            xmin = float(bounding_box.find('xmin').text)/width
            xmax = float(bounding_box.find('xmax').text)/width
            ymin = float(bounding_box.find('ymin').text)/height
            ymax = float(bounding_box.find('ymax').text)/height
            class_label =  bbox.find('name').text
            label_data.append([image_file, width, height, class_label, xmin, ymin, xmax, ymax])
	columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    bbox_df = pd.DataFrame(data=label_data, columns=columns)
    return bbox_df, image_files_list, label_files_list


# Function Name: compute_IOU_score
# Input        : ground truth and predicted bounding box.
# Output       : Return the IoU
# Description  : This method computes IoU between predicted and ground truth.
def compute_IOU_score(actual, predicted):
    ac_xmin, ac_ymin, ac_xmax, ac_ymax = actual
    pr_ymin, pr_xmin, pr_ymax, pr_xmax = predicted
    
    # calculate the intersection box coodinates
    xmin = 0
    ymin = 0 
    # (xmin, ymin) are the top left corner, 
	# so taking the max to get intersection coordinate
    if ac_xmin >= pr_xmin  and  ac_xmin <= pr_xmax:
        xmin = ac_xmin
    elif pr_xmin >= ac_xmin  and  pr_xmin <= ac_xmax:
        xmin = pr_xmin
    if ac_ymin >= pr_ymin  and  ac_ymin <= pr_ymax:
        ymin = ac_ymin
    elif pr_ymin >= ac_ymin  and  pr_ymin <= ac_ymax:
        ymin = pr_ymin    
    
    
    # (xmax, ymac) are the bottom right corner, 
	# Hence taking the min to get intersection coordinate
    xmax = 0
    ymax = 0 
    
    if ac_xmax <= pr_xmax and ac_xmax >= pr_xmin:
        xmax = ac_xmax
    elif pr_xmax <= ac_xmax and pr_xmax >= ac_xmin:
        xmax = pr_xmax
    if ac_ymax <= pr_ymax and ac_ymax >= pr_ymin:
        ymax = ac_ymax
    elif pr_ymax <= ac_ymax and pr_ymax >= ac_ymin:
        ymax = pr_ymax
  
    
    # Area = height * width
    InterArea = (xmax - xmin)*(ymax - ymin)
    
    # Computing Union Area
    ActualBox = (ac_xmax - ac_xmin) * (ac_ymax - ac_ymin)
    PredBox = (pr_xmax - pr_xmin) * (pr_ymax - pr_ymin)
    
    IoU = 0
    
    if InterArea != 0:
        IoU = InterArea/float(ActualBox + PredBox - InterArea)
    return IoU

# Function Name: determine_boundary
# Input        : input image in the form of numpy array.
# Output       : Return the boundary coordinates of the image
# Description  : This method will calculate the actual boundary of the image
# Invoked by   : check_for_partial_objects
def determine_boundary(img):
    i = 0
    result = True
    while result == True:
        pixel = list(img[i])
        result = pixel.count(pixel[0]) == len(pixel)
        if (result):
            i = i+1
    ymin = i+1
    result = True
    i = img.shape[1]-1 #height
    while result == True:
        pixel = list(img[i])
        result = pixel.count(pixel[0]) == len(pixel)
        if (result):
            i = i-1
    ymax = i
    
    img = img.T
    i = 0
    result = True
    while result == True:
        pixel = list(img[i])
        result = pixel.count(pixel[0]) == len(pixel)
        if (result):
            i = i+1
    xmin = i+1
    result = True
    i = img.shape[1]-1 #height
    while result == True:
        pixel = list(img[i])
        result = pixel.count(pixel[0]) == len(pixel)
        if (result):
            i = i-1
    xmax = i
    return xmin, xmax, ymin, ymax

# Function Name: determine_partial_circle
# Input        : input image in the form of numpy array.
# Output       : Return whether the detection is partial object or not
#                indicate it is a partial detection if the object 
#                has a actual dimension less that 0.6 of the detected 
#                radius 
# Invoked by   : check_for_partial_objects
def determine_partial_circle(input_img, left, right, up, down):
    gray_crop = input_img
    dimension = input_img.shape[0]
    if input_img.shape[1] > dimension:
        dimension = input_img.shape[1]
    minRad = int(dimension * 0.25)
    maxRad = dimension
    circles = cv.HoughCircles(gray_crop, 				
							  cv.HOUGH_GRADIENT, 		
							  1, 						
							  100, 						
							  param1=50, 				
							  param2=12, 					
							  minRadius=minRad, 		
							  maxRadius=maxRad)
    partial_detection = False
    if circles is not None:
        detected_circles = np.uint16(np.around(circles))

        for (x, y ,r) in detected_circles[0, :]:
            height = gray_crop.shape[0]
            width = gray_crop.shape[1]
            if left == True and (x <= (0.6 * r)):
                partial_detection = True
            if right == True and ((width - x) <= (0.6 * r)):
                partial_detection = True
            if up == True and (y <= (0.6 * r)):
                partial_detection = True
            if down == True and ((height - y) <= (0.6 * r)):
                partial_detection = True                
    else:
        partial_detection = True
    return partial_detection

# Function Name: check_for_partial_objects
# Input        : input_img - input image in the form of numpy array.
#                bbox - detected box
# Output       : Return whether the detection is partial object or not
# Description  : This method will check if the predicted box falls on 
#                image boundary. if so invokes determine_partial_circle
#Invoked by    : remove_partial_detections
def check_for_partial_objects(input_img, bbox):
    imag = input_img
    gray = cv.cvtColor(imag, cv.COLOR_BGR2GRAY)
    xmin, xmax, ymin, ymax = determine_boundary(gray)
    canny = cv.Canny(gray,25,50)
    (T, gray) = cv.threshold(canny, 128, 255, cv.THRESH_BINARY)
    bymin = int(bbox[0] * input_img.shape[1])
    bxmin = int(bbox[1] * input_img.shape[0])
    bymax = int(bbox[2] * input_img.shape[1])
    bxmax = int(bbox[3] * input_img.shape[0])
    left = False
    right = False
    up = False
    down = False
    if bymin > ymin:
        bymin = bymin -1
    if bymax < ymax -1:
        bymax = bymax + 1  
    if bxmin > xmin:
        bxmin = bxmin -1
    if bxmax < xmax -1:
        bxmax = bxmax + 1
    if bxmin <= xmin:
        left = True  
    if bxmax >= xmax:
        right = True
    if bymin <= ymin:
        up = True  
    if bymax >= ymax:
        down = True       
    gray_crop = gray[bymin:bymax,bxmin:bxmax]
    partial = False
    if left== True or right==True or up==True or down==True:
        partial = determine_partial_circle(gray_crop, left, right, up, down)
    return partial;

# Function Name: detect_duplicate_boxes
# Input        : det_boxs - detected boxes in decreasing confidence score
#                det_class_category - detected classes
# Output       : Return the list of det_boxs, det_class_category after duplicate removal
# Description  : This method will compute IoU  of each detection with
#                all other boxes having higher score. If the IoU > 0.85 the detection 
#                is duplicate and is removed from final list.
def detect_duplicate_boxes(det_boxs, det_class_category):
  filetered_box = []
  filtered_class_category = []
  filetered_box.append(det_boxs[0])
  filtered_class_category.append(det_class_category[0])
  for i in range(1, len(det_boxs)):
    pr_ymin, pr_xmin, pr_ymax, pr_xmax = det_boxs[i]
    current_box = [pr_xmin, pr_ymin, pr_xmax, pr_ymax]
    duplicate = False
    for j in range(len(filetered_box)):
      curIoU = compute_IOU_score(current_box, filetered_box[j])
      if curIoU > 0.85:
        duplicate = True
        break;
    if duplicate == False:
      filetered_box.append(det_boxs[i])
      filtered_class_category.append(det_class_category[i])
  return filetered_box, filtered_class_category

# Function Name: remove_partial_detections
# Input        : input_img - input image in the form of numpy array.
#                det_boxs - detected boxes in decreasing confidence score
#                det_class_category - detected classes
# Output       : Return he list of det_boxs, det_class_category after partial object removal
def remove_partial_detections(image, det_boxs, det_class_category):
  filetered_box = []
  filtered_class_category = []
  for i in range(len(det_boxs)):
    partial_detection = False
    partial_detection = check_for_partial_objects(image, det_boxs[i])
    if partial_detection == False:
      filetered_box.append(det_boxs[i])
      filtered_class_category.append(det_class_category[i])
  return filetered_box, filtered_class_category

# Function Name: create_confusion_matrix
# Input        : prediction_data - Dataframe contains details on TP, FP, FN
# Output       : Return Confusion Matrix DataFrame
def create_confusion_matrix(prediction_data):
  Total_TP_Agg = prediction_data.TP_Aggregate.sum()
  Total_FP_Aggregate_TP_Punctate = prediction_data.FP_Aggregate_TP_Punctate.sum()
  Total_FP_Aggregate_TP_Erythrocyte = prediction_data.FP_Aggregate_TP_Erythrocyte.sum()
  Total_FP_Aggregate_extra = prediction_data.FP_Aggregate.sum()
  Total_TP_Punc = prediction_data.TP_Punctate.sum()
  Total_FP_Punc_TP_Aggregate = prediction_data.FP_Punctate_TP_Aggregate.sum()
  Total_FP_Punc_TP_Erythrocyte = prediction_data.FP_Punctate_TP_Erythrocyte.sum()
  Total_FP_Punc_extra = prediction_data.FP_Punctate.sum()
  Total_TP_Ery = prediction_data.TP_Erythrocyte.sum()
  Total_FP_Ery_TP_Aggregate = prediction_data.FP_Erythrocyte_TP_Aggregate.sum()
  Total_FP_Ery_TP_Punctate =prediction_data.FP_Erythrocyte_TP_Punctate.sum()
  Total_FP_Ery_Extra = prediction_data.FP_Erythrocyte.sum()
  confusion_matrix = []
  confusion_matrix.append([Total_TP_Agg,
                          Total_FP_Aggregate_TP_Punctate,
                          Total_FP_Aggregate_TP_Erythrocyte,
                          Total_FP_Aggregate_extra,
                          prediction_data.FN_Aggregate.sum()])
  confusion_matrix.append([Total_FP_Punc_TP_Aggregate,
                          Total_TP_Punc,
                          Total_FP_Punc_TP_Erythrocyte,
                          Total_FP_Punc_extra,
                          prediction_data.FN_Punctate.sum()])
  confusion_matrix.append([Total_FP_Ery_TP_Aggregate,
                          Total_FP_Ery_TP_Punctate,
                          Total_TP_Ery,
                          Total_FP_Ery_Extra,
                          prediction_data.FN_Erythrocyte.sum()])
  confusion_df = pd.DataFrame(confusion_matrix, columns = ["Aggregate", "Punctate", "Erythrocyte", "FP", "FN"])
  return confusion_matrix, confusion_df

# Function Name: compute_accuracy
# Input        : confusion_matrix - Confusion matrix as list
# Output       : Return the accuracy
def compute_accuracy(confusion_matrix):
  TP = confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2]
  FALSE_DET = confusion_matrix[0][1] + confusion_matrix[0][2] +\
  confusion_matrix[1][0] + confusion_matrix[1][2] +\
  confusion_matrix[2][0] + confusion_matrix[2][1] +\
  confusion_matrix[0][3] + confusion_matrix[0][4] +\
  confusion_matrix[1][3] + confusion_matrix[1][4] +\
  confusion_matrix[2][3] + confusion_matrix[2][4]

  Accuracy = TP/(TP+FALSE_DET)

  return Accuracy


        