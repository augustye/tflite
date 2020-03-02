import time
import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt
from pysenet.estimator import *

model_path = "model/multi_person_mobilenet_v1_075_float.tflite"

stride = 16
threshold = 0.5
nmsr = 20
radius = 1
detection = 5

estimator = Estimator(stride,model_path,threshold,detection,nmsr,radius)

test_img = "images/pose3.jpg"
image = cv2.imread(test_img)

img_transposed = np.zeros(image.shape,"uint8");
img_transposed[:,:,0] = image[:,:,2]
img_transposed[:,:,1] = image[:,:,1]
img_transposed[:,:,2] = image[:,:,0]
image = img_transposed

image = crop_image(image)

(fac,fitted_img) = resize_img(image)

poses = estimator.process_img(fitted_img,fac)
print("poses:", poses)

estimator.draw_pose_with_ease(max(poses),image)
#estimator.draw_poses(poses,image,-100)
cv2.imwrite("test.jpg", image)