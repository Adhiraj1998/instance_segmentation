import numpy as np
import argparse
import random
import time
import cv2
import os


seg_ip = 'D:/datasets/camvid/instance/seg/'
roi_ip = 'D:/datasets/camvid/instance/roi/'

seg_op = 'D:/datasets/camvid/instance/seg_resize/'
roi_op = 'D:/datasets/camvid/instance/roi_resize/'

lis_ip = [seg_ip, roi_ip]
lis_op = [seg_op, roi_op]

for directory in lis_op:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass

ap = argparse.ArgumentParser()
ap.add_argument("-h", "--height", required=True,
	help="desired height")

ap.add_argument("-w", "--width", required=True,
	help="desired width")
args = vars(ap.parse_args())    

for i in lis_ip:

    for j in os.listdir(i):
        print("Image Name", j)
        image = cv2.imread(os.path.join(i, j))
        width = args["width"]
        height = args["height"]
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        if i is seg_ip:
            name = seg_op+j+'.jpg'
        else:
            name = roi_op+j+'.jpg'
        cv2.imwrite(name, resized)


