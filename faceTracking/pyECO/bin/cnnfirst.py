import os
import numpy as np
import sys
from PIL import Image
import dlib

cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
filename = "../../../test/0001.jpg"
img = dlib.load_rgb_image(filename)
dets = cnn_face_detector(img, 1)
print(dets[0].rect)
#bbox = [dets[0].rect.left()+1,dets[0].rect.top()+1,dets[0].rect.right()-dets[0].rect.left(),dets[0].rect.bottom()-dets[0].rect.top()]
