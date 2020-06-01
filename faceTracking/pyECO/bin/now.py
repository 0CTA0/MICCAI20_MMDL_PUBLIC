import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import sys
import dlib
sys.path.append('./')
from eco import ECOTracker
from PIL import Image

def main():
    # load videos
    filenames = sorted(glob.glob("../../test/*.jpg"),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    frames = [np.array(Image.open(filename)) for filename in filenames]
    height, width = frames[0].shape[:2]
    # starting tracking
    tracker = ECOTracker(True)
    vis = True
    videoWriter = cv2.VideoWriter('0011.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (224,224))
    for idx, frame in enumerate(frames):
        if idx == 0:
            bbox = [245, 512,587,587] #[dets[0].left()+1,dets[0].top()+1,dets[0].right()-dets[0].left(),dets[0].bottom()-dets[0].top()]
            tracker.init(frame, bbox)
            bbox = (bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
        elif idx < len(frames) - 1:
            bbox = tracker.update(frame, True, vis)
        else: # last frame
            bbox = tracker.update(frame, False, vis)
        # bbox xmin ymin xmax ymax
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if len(bbox)==4:
            print("write "+str(idx))
            actualbbox =[max(0,int(bbox[1])),min(int(bbox[3]),frame.shape[0]),max(0,int(bbox[0])),min(int(bbox[2]),frame.shape[1])]
            saveframe = frame[actualbbox[0]:actualbbox[1],actualbbox[2]:actualbbox[3]]
            longerside = max(actualbbox[1]-actualbbox[0],actualbbox[3]-actualbbox[2])
            dstsize = (224*(actualbbox[1]-actualbbox[0])/longerside,224*(actualbbox[3]-actualbbox[2])/longerside)
            dstsize = (int(dstsize[1]),int(dstsize[0]))
            resizedimg = cv2.resize(saveframe,dstsize,interpolation = cv2.INTER_AREA)
            canvas = 255*np.ones((224,224,3), np.uint8)
            try:
                canvas[max(0,-int(int(bbox[1])*224/longerside)):(max(0,-int(int(bbox[1])*224/longerside))+dstsize[1]),max(0,-int(int(bbox[0])*224/longerside)):(max(0,-int(int(bbox[0])*224/longerside))+dstsize[0])] = resizedimg
                videoWriter.write(canvas)
            except:
                continue
    videoWriter.release()
if __name__ == "__main__":
    main()