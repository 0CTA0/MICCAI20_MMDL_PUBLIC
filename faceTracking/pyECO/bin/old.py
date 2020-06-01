import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import sys
sys.path.append('./')
from eco import ECOTracker
from PIL import Image
import dlib
import argparse

def main(video_dir):
    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    frames = [np.array(Image.open(filename)) for filename in filenames]
    height, width = frames[0].shape[:2]
    if len(frames[0].shape) == 3:
        is_color = True
    else:
        is_color = False
        frames = [frame[:, :, np.newaxis] for frame in frames]
    # starting tracking
    tracker = ECOTracker(is_color)
    vis = True
    for idx, frame in enumerate(frames):
        print(idx)
        if idx == 0:
            cnn_face_detector = dlib.cnn_face_detection_model_v1("bin/mmod_human_face_detector.dat")
            filename = os.path.join(video_dir, "0001.jpg")
            img = dlib.load_rgb_image(filename)
            dets = cnn_face_detector(img, 1)
            bbox = [dets[0].rect.left()+1,dets[0].rect.top()+1,dets[0].rect.right()-dets[0].rect.left(),dets[0].rect.bottom()-dets[0].rect.top()]
            tracker.init(frame, bbox)
            bbox = (bbox[0]-1, bbox[1]-1,
                    bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)
        elif idx < len(frames) - 1:
            bbox = tracker.update(frame, True, vis)
        else: # last frame
            bbox = tracker.update(frame, False, vis)
        # bbox xmin ymin xmax ymax
        frame = frame.squeeze()
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.rectangle(frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 255),
                              1)
        frame = frame.squeeze()
        if vis and idx > 0:
            score = tracker.score
            size = tuple(tracker.crop_size.astype(np.int32))
            score = cv2.resize(score, size)
            score -= score.min()
            score /= score.max()
            score = (score * 255).astype(np.uint8)
            # score = 255 - score
            score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
            pos = tracker._pos
            pos = (int(pos[0]), int(pos[1]))
            xmin = pos[1] - size[1]//2
            xmax = pos[1] + size[1]//2 + size[1] % 2
            ymin = pos[0] - size[0] // 2
            ymax = pos[0] + size[0] // 2 + size[0] % 2
            left = abs(xmin) if xmin < 0 else 0
            xmin = 0 if xmin < 0 else xmin
            right = width - xmax
            xmax = width if right < 0 else xmax
            right = size[1] + right if right < 0 else size[1]
            top = abs(ymin) if ymin < 0 else 0
            ymin = 0 if ymin < 0 else ymin
            down = height - ymax
            ymax = height if down < 0 else ymax
            down = size[0] + down if down < 0 else size[0]
            score = score[top:down, left:right]
            crop_img = frame[ymin:ymax, xmin:xmax]
            score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
            frame[ymin:ymax, xmin:xmax] = score_map
        saveFileName = "new%04d" % idx
        cv2.imwrite(saveFileName + ".jpg",frame)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', type=str, default='sequences/Crossing/')
    args = parser.parse_args()
    main(args.v)
