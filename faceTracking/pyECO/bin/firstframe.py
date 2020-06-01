import numpy as np
import cv2
img = cv2.imread('../../test/0001.jpg', 1)
print(img.shape)
img = cv2.rectangle(img,
                        (244, 511),
                        (831, 1098),
                        (0, 255, 255),
                        1)
cv2.imwrite("first.jpg",img)
