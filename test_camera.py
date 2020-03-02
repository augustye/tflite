import time
import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt
from pysenet.estimator import *

model_path = "model/multi_person_mobilenet_v1_075_float.tflite"
estimator = Estimator(16,model_path,0.5,5,50,1)
cam = cv2.VideoCapture(0)

start_time = time.time()
read_time = 0
resize_time = 0
predict_time = 0
draw_time = 0
frames = 0
while True:
    read_start = time.time()
    ret_val, image = cam.read()
    read_time += time.time() - read_start

    resize_start = time.time()
    image = crop_image(image)
    (fac,fitted_img) = resize_img(image)
    resize_time += time.time() - resize_start

    predict_start = time.time()
    poses = estimator.process_img(fitted_img,fac)
    predict_time += time.time() - predict_start

    draw_start = time.time()
    estimator.draw_poses(poses,image,-100)
    cv2.imshow('cam_demo',image)
    draw_time += time.time() - draw_start

    frames += 1

    if frames%100 == 0:
        print("frames: %d, fps:%d, read_time:%f, resize_time:%f, predict_time:%f, draw_time:%f"
            %(frames,frames/(time.time()-start_time),read_time,resize_time,predict_time,draw_time))

    #if cv2.waitKey(1) == 27:
    #    break

cv2.destroyAllWindows()