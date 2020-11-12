#This file takes images from a particular directory as input and uses the MaskRCNN algorithm pre-trained on MS-COCO weights to implement instance segmentation. 
# We require the representation of the output in the form of the bounding box image of the objects that have been detected and the segmented images based on the mask we have used to implement the task.

import numpy as np
import argparse
import random
import time
import cv2
import os

#Directories of the path to the input images and the output directories for the main output, segmented images and region of interest
path = './images/'
seg_p = './segmented/'
roi_p = './roi/'
output_p = './output/'

#Create the directories if they don't exist
lis = [seg_p, roi_p, output_p]

for directory in lis:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass

#Command line parsing of arguments. Defaults have been set so unless there is a specific parameter you wish to edit, you may do so in the command line
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--mask-rcnn", default = './mask-rcnn-coco',
	help="path to Mask RCNN weights and class ids")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="lower limit of acceptable detections")
ap.add_argument("-t", "--threshold", type=float, default=0.7,
	help="lower limit for pixel-based image segmentation")
args = vars(ap.parse_args())

# We need to load the class labels based on which the MaskRCNN was trained on
path_labels = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(path_labels).read().strip().split("\n")

path_colours = os.path.sep.join([args["mask_rcnn"], "colours.txt"])
colours = open(path_colours).read().strip().split("\n")

colours = [np.array(c.split(",")).astype("int") for c in colours]
colours = np.array(colours, dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"weights.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2.pbtxt"])

print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

#Setting a progressive numbering system of saving the output files
a = 1
b = 1
c = 1

for i in os.listdir(path):
    print("Image name", i)
    image = cv2.imread(os.path.join(path, i))

    (H, W) = image.shape[:2]
    #Create a blob for the sample image
    # A blob is Binary Large Object and refers to the connected pixel in the binary image. It is a 4 dimensional vector. 
    #Changes that are needed to be made: swap R and B channels as OpenCV loads the image as RBG instead of RGB.
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    
    (out, dm) = net.forward(["detection_out_final", "detection_masks"])
    for i in range(0, out.shape[2]):
        classID = int(out[0, 0, i, 1]) #label ID of the detected object
        confidence = out[0, 0, i, 2] #probability of detected object being the alloted ID. the ID with the highest confidence is assigned to the output
        
        if confidence > args["confidence"]: #We do not want low probability outputs. The threshold can be changed in the command line
            clone = image.copy()
            b_array = np.array([W, H, W, H])
            box = out[0, 0, i, 3:7] * b_array #Setting the bounding box
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            mask = dm[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_AREA)
            mask = (mask > args["threshold"])

            # extract the Region of Interest (RoI) or the bounding box for the object detection
            roi = clone[startY:endY, startX:endX]
     
            visMask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(roi, roi, mask=visMask)
            
            #Sometimes erronous objects get detected an occupy a strange and small part of the input image. These anomalies usually bypass the confidence threshold by having very high probabilities.
            # We set a threshold of 5000 to remove these anomalies.
            row, col = len(roi), len(roi[0])
            area = row*col

            if area < 10000:
                continue 
            else:
                name_a = roi_p+'img'+str(a)+'.jpg'
                name_b = seg_p+'img'+str(b)+'.jpg'
                name_c = output_p+'img'+str(c)+'.jpg'
                
                #To save to and update the RoI directory
                cv2.imwrite(name_a, roi)
                a = a+1
                
                #To save to and update the segmented directory
                cv2.imwrite(name_b, instance)
                b = b+1
                    
                roi = roi[mask]

                #choose a random coloour from the colour list
                colour = random.choice(colours)
                blended = ((0.4 * colour) + (0.6 * roi)).astype("uint8")

                # store the blended ROI in the original image
                clone[startY:endY, startX:endX][mask] = blended
                colour = [int(c) for c in colour]
                cv2.rectangle(clone, (startX, startY), (endX, endY), colour, 2)
                text = "{}: {:.4f}".format(LABELS[classID], confidence)
                print("DETECTED ", LABELS[classID], "  ", confidence)
                cv2.putText(clone, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

                # Save the output image
                cv2.imwrite(name_c, clone)
                c = c+1
    


