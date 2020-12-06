# Instance Segmentation
This repository deals with the task of implementing instance segmentation for a given dataset using Mask-RCNN architecture.

The dataset is provided [here](https://drive.google.com/drive/folders/1FGRxP54FNXJvWc-vtkHSLACeqqoY4OZc?usp=sharing). Add these images to the ./image folder in the repository or provide the path of the repository in the code. Currently it contains 100 images taken from the Dogs vs. Cats dataset. Feel free to replace it with other images.


## Install Dependencies
Run the following to ensure that all the dependencies needed for executing the code is present:

    pip install -r requirements.txt

## Directory
instance.py is the main document to be executed. The code contains 4 parameters that may be defined in the command line. 

1. input: path to dataset directory. Default is ./images/, the image folder in the current working directory 
2. mask-rcnn: path to weights, labels and configuration folder for Mask RCNN, trained on COCO dataset. Default is ./mask-rcnn-coco, located in the current working directory.
3. confidence: threshold for detections. Deafult is 0.7
4. threshold: threshold for pixel-based segmentation. Default is 0.5

## Execution 
To run the code with the default parameters, run the following in the directory's command line:
	python instance.py

To run the code with user-defined parameters, run the following in the directory's command line and fill the appropriate value for each parameter:
	python instance.py --mask-rcnn [PATH TO MASK-RCNN-COCO FOLDER] --input [PATH TO IMAGES] --confidence [CONFIDENCE VALUE TO BE SET] --threshold [THRESHOLD VALUE TO BE SET]

## Output
You will find three folders formed in the current working directory. They are segmented, roi and output. 
Segmented and roi contain the segmented object in each image and the cropped bounding box of the detected objects respecitvely. They serve different purposes moving forward.
output shows the instance detections of objects on the original image.

