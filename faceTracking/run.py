# from glob import glob
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import os
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from torchvision import transforms
from filterpy.kalman import KalmanFilter
import torch
from vidstab.VidStab import VidStab
from filterpy.common import Q_discrete_white_noise
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from torch.autograd import Variable
from imutils.video import FileVideoStream
from imutils import face_utils
from eco import ECOTracker

ROTATION_ANGLE = 90
device = torch.device("cpu")

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

class FaceDetector():
    def __init__(self, cascPath="./haarcascade_frontalface_default.xml"):
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        rects = detector(gray, 0)
        for face in rects:
            faces.append([face.left(), face.top(), face.width(), face.height()])
        return faces

class FaceTracker():
    
    def __init__(self, frame, face):
        (x,y,w,h) = face
        self.face = (x,y,w,h)
        # Arbitrarily picked KCF tracking
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.face)
    
    def update(self, frame):
        _, self.face = self.tracker.update(frame)
        return self.face

class Controller():
    
    def __init__(self, event_interval=6):
        self.event_interval = event_interval
        self.last_event = datetime.datetime.now()

    def trigger(self):
        """Return True if should trigger event"""
        return self.get_seconds_since() > self.event_interval
    
    def get_seconds_since(self):
        current = datetime.datetime.now()
        seconds = (current - self.last_event).seconds
        return seconds

    def reset(self):
        self.last_event = datetime.datetime.now()

class Pipeline():

    def __init__(self, event_interval=6):
        self.controller = Controller(event_interval=event_interval)    
        self.detector = FaceDetector()
        self.trackers = []
    
    def detect_and_track(self, frame):
        faces = self.detector.detect(frame)
        self.controller.reset()
        self.trackers = [FaceTracker(frame, face) for face in faces]
        new = len(faces) > 0

        return faces, new
    
    def track(self, frame):
        boxes = [t.update(frame) for t in self.trackers]
        return boxes, False
    
    def boxes_for_frame(self, frame):
        return self.track(frame)

def cal_dist_diff(curPoints,lastPoints):
    variance = 0.0
    sum_ = 0.0
    diffs = []
    if curPoints.shape == lastPoints.shape:
        for i in range(curPoints.shape[0]):
            diff = math.sqrt(pow(curPoints[i][0] - lastPoints[i][0], 2.0) + pow(curPoints[i][1] - lastPoints[i][1], 2.0))
            sum_ += diff
            diffs.append(diff)
        
        mean = sum_ / len(diffs)
        for i in range(curPoints.shape[0]):
            variance += pow(diffs[i] - mean, 2)
        
        return variance / len(diffs)
    return variance

def func(x, a, b):
    return a*x + b

def calc_pts(curPoints,lastPoints):
    diff = 0. 
    for i in range(17):
        temp = math.sqrt(pow(curPoints[i][0] - lastPoints[i][0], 2.0) + pow(curPoints[i][1] - lastPoints[i][1], 2.0))
        diff += math.sqrt(temp)
        # diffs.append(diff)
    return diff


landmarks_3d_list = [
    np.array([
        [ 0.000,  0.000,   0.000],    # Nose tip
        [ 0.000, -8.250,  -1.625],    # Chin
        [-5.625,  4.250,  -3.375],    # Left eye left corner
        [ 5.625,  4.250,  -3.375],    # Right eye right corner
        [-3.750, -3.750,  -3.125],    # Left Mouth corner
        [ 3.750, -3.750,  -3.125]     # Right mouth corner 
    ], dtype=np.double),
    np.array([
        [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
        [ 6.825897,  6.760612,  4.402142],   # 33 left brow left corner
        [ 1.330353,  7.122144,  6.903745],   # 29 left brow right corner
        [-1.330353,  7.122144,  6.903745],   # 34 right brow left corner
        [-6.825897,  6.760612,  4.402142],   # 38 right brow right corner
        [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
        [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
        [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
        [-5.311432,  5.485328,  3.987654],   # 21 right eye right corner
        [ 2.005628,  1.409845,  6.165652],   # 55 nose left corner
        [-2.005628,  1.409845,  6.165652],   # 49 nose right corner
        [ 2.774015, -2.080775,  5.048531],   # 43 mouth left corner
        [-2.774015, -2.080775,  5.048531],   # 39 mouth right corner
        [ 0.000000, -3.116408,  6.097667],   # 45 mouth central bottom corner
        [ 0.000000, -7.415691,  4.070434]    # 6 chin corner
    ], dtype=np.double),
    np.array([
        [ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
        [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
        [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
        [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
        [-5.311432,  5.485328,  3.987654]    # 21 right eye right corner
    ], dtype=np.double)
]

# 2d facial landmark list
lm_2d_index_list = [
    [30, 8, 36, 45, 48, 54],
    [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8], # 14 points
    [33, 36, 39, 42, 45] # 5 points
]
lm_2d_index = lm_2d_index_list[1]
landmarks_3d = landmarks_3d_list[1]

def to_numpy(landmarks):
        coords = []
        for i in lm_2d_index:
            coords += [[landmarks[i][0], landmarks[i][1]]]
        return np.array(coords).astype(np.double)





if __name__ == '__main__':
    face_locations = []

    a = argparse.ArgumentParser()
    a.add_argument("--Videopath","-v")
    args = vars(a.parse_args())
    file_path = "../RawData/Video/"+args["Videopath"]+".MOV"
    print("Initializing Face Segmentation...")
    fvs = FileVideoStream(file_path).start()
    img_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)])
    t = transforms.Resize(size=(256,256))
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    fvs = FileVideoStream(file_path).start()
    stabilizer = VidStab()
    prev_anchor = None
    prev_ROTATION = None
    MOTION_DIFF = 0
    frame_count = 0
    refreshed = False
    # ROTATION_ANGLE = 75
    tracker = ECOTracker(True)

    prevTrackPts = []
    nextTrackPts = []
    last_object = []
    kalman_points = []
    predict_points = []
    for i in range(68):
        predict_points.append((0.0, 0.0))
        last_object.append((0.0, 0.0))
        kalman_points.append((0.0, 0.0))
        prevTrackPts.append((0.0, 0.0))
        nextTrackPts.append((0.0, 0.0))

    scaling = 0.5
    flag = -1
    count = 0
    redetected = True
    stateNum = 272
    measureNum = 136

    kalman = cv2.KalmanFilter(stateNum, measureNum, 0)
    state = 0.1 * np.random.randn(stateNum,1)
    processNoise = np.zeros((stateNum, 1))
    measurement = np.zeros((measureNum, 1))
    kalman.transitionMatrix = np.zeros((stateNum,stateNum))
    
    for i in range(stateNum):
        for j in range(stateNum):
            if i == j or (j - measureNum) == i:
                kalman.transitionMatrix[i, j] = 1.0
            else:
                kalman.transitionMatrix[i, j] = 0.0
               
    prevgray = None
    gray = None
    kalman.measurementMatrix = 1. * np.eye(measureNum, stateNum)
    kalman.processNoiseCov = 1e-5 * np.eye(stateNum)
    kalman.measurementNoiseCov = 1e-1 * np.eye(measureNum, measureNum)
    kalman.errorCovPost = np.ones((stateNum, stateNum))
    kalman.statePost = 0.1 * np.random.randn(stateNum,1)

    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    
    prev_rect = None
    ROTATION_ANGLE = 90
    is_frame = True
    vid_count = 0
    
    THRES = 150
    faces = ()
    detected = False
    cap = cv2.VideoCapture(file_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pipeline = Pipeline(length)
    detected = False
    satisfied = False
    video = cv2.VideoWriter("../Feature/Crop/"+args["Videopath"]+".avi", cv2.VideoWriter_fourcc(*'MP42'),30, (224,224))

    while not detected:
        frame = fvs.read()
        
        frame = imutils.rotate_bound(frame,ROTATION_ANGLE)
        frame = imutils.resize(frame, width=800)
        faces, detected = pipeline.detect_and_track(frame)
        if not len(faces) or faces[0][0] <0:
            detected = False
            continue
        print("hot start; ", faces, type(faces), "size: ", np.array(faces).size)
        initbox = faces[0]
        tracker.init(frame,faces[0])

    while is_frame:
        frame_count = 0
        tf = 0
        diffs = []
        rys = []
        drys = []
        prx,pry,prz = 0,0,0
        flag = -1
        while True:
            if tf > 30:
                flag = -1
                tf = 15
            frame0 = fvs.read()
            if frame0 is None:
                is_frame = False
                break
            tf += 1
            frame = imutils.rotate_bound(frame0,ROTATION_ANGLE)
            frame = imutils.resize(frame, width=800)
            boxes = tracker.update(frame,True,True)
                    
            frame0 = imutils.resize(frame, width=360)
            lms = []
            face = frame0
            gr =  cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            rect = detector(gr, 0)
            
            if flag == -1:
                if not (len(rect)):
                    continue
                prevgray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                shape = predictor(gr, rect[0])
                preds = face_utils.shape_to_np(shape)
                for i,pts in enumerate(preds):
                    prevTrackPts[i] = (pts[0],pts[1])
            if len(rect):
                shape = predictor(gr, rect[0])
                preds = face_utils.shape_to_np(shape)
                for i,pts in enumerate(preds):
                    kalman_points[i] = (pts[0],pts[1])
            else:
                shape = predictor(gr, prev_rect[0])
                preds = face_utils.shape_to_np(shape)
                for i,pts in enumerate(preds):
                    kalman_points[i] = (pts[0],pts[1])
            # print(kalman_points)

            prediction = kalman.predict()
            # // std::vector<cv::Point2f> predict_points;
            for i in range(68):
                predict_points[i] = (prediction[2*i][0],prediction[i * 2 + 1][0])
            gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            diff = 0.
            if len(rect):
                if prevgray is not None:
                    nextTrackPts, status, err = cv2.calcOpticalFlowPyrLK(prevgray, gray, np.float32(prevTrackPts), np.float32(nextTrackPts))
                    
                    diff = cal_dist_diff(np.array(prevTrackPts), np.array(nextTrackPts))
                    # print("variance:",diff)
                    if flag == -1 or diff > 3 or tf < 15 :
                        flag = 1
                        shape = predictor(gr, rect[0])
                        # print("DLIB")
                        preds = face_utils.shape_to_np(shape)
                        lms = preds
                        for i,pts in enumerate(preds):
                            nextTrackPts[i] = (pts[0],pts[1])
                    elif diff <= 3 and diff > 0.0005:
                        # print("Optical Flow")
                        lms = nextTrackPts
                    else:
                        # print("Kalman Filter")
                        lms = predict_points
                        for i,pts in enumerate(predict_points):
                            nextTrackPts[i] = (pts[0],pts[1])
                        redetected = False
                else:
                    redetected = True
                
                # print(predict_points)
                tt = prevTrackPts
                prevTrackPts = nextTrackPts
                nextTrackPts = tt

                tg = prevgray
                prevgray = gray
                gray = tg
            else:
                if prevgray is not None:
                    nextTrackPts, status, err = cv2.calcOpticalFlowPyrLK(prevgray, gray, np.float32(prevTrackPts), np.float32(nextTrackPts))
                    
                    diff = cal_dist_diff(np.array(prevTrackPts), np.array(nextTrackPts))
                    # print("variance:",diff)
                    if flag == -1 or diff > 3 or tf < 15 :
                        flag = 1
                        shape = predictor(gr, prev_rect[0])
                        # print("DLIB")
                        preds = face_utils.shape_to_np(shape)
                        lms = preds
                        for i,pts in enumerate(preds):
                            nextTrackPts[i] = (pts[0],pts[1])
                    elif diff <= 3 and diff > 0.0005:
                        # print("Optical Flow")
                        lms = nextTrackPts
                    else:
                        # print("Kalman Filter")
                        lms = predict_points
                        for i,pts in enumerate(predict_points):
                            nextTrackPts[i] = (pts[0],pts[1])
                        redetected = False
                else:
                    redetected = True
                
                # print(predict_points)
                tt = prevTrackPts
                prevTrackPts = nextTrackPts
                nextTrackPts = tt

                tg = prevgray
                prevgray = gray
                gray = tg
            
            # print(len(kalman_points))
            for i in range(136):
                if i % 2 == 0:
                    measurement[i] = kalman_points[int(i / 2)][0]
                else:
                    measurement[i] = kalman_points[int((i - 1) / 2)][1]
            measurement += kalman.measurementMatrix.dot(state)
            # print(kalman.measurementMatrix)
            kalman.correct(measurement)

            if len(rect):
                prev_rect = rect
            if diff < 0.01:
                # print("small change")
                continue
                        
            h, w, c = frame0.shape
            f = w # column size = x axis length (focal length)
            u0, v0 = w / 2, h / 2 # center of image plane
            camera_matrix = np.array(
                [[f, 0, u0],
                [0, f, v0],
                [0, 0, 1]], dtype = np.double
            )
            
            # Assuming no lens distortion
            dist_coeffs = np.zeros((4,1)) 
            landmarks_2d = to_numpy(lms)
            # Find rotation, translation
            (success, rotation_vector, translation_vector) = cv2.solvePnP(landmarks_3d, landmarks_2d, camera_matrix, dist_coeffs)
            
            # return rotation_vector, translation_vector, camera_matrix, dist_coeffs
            rmat = cv2.Rodrigues(rotation_vector)[0]
            P = np.hstack((rmat, translation_vector)) # projection matrix [R | t]
            degrees = -cv2.decomposeProjectionMatrix(P)[6]
            rx, ry, rz = degrees[:, 0]
            diffs.append(calc_pts(prevTrackPts,nextTrackPts))
            if len(diffs) == 6:
                diffs.pop(0)
            drys.append(abs(ry-pry))
            if len(drys) == 6:
                drys.pop(0)
            
            rys.append(abs(ry))
            if len(rys) == 6:
                rys.sort()
                rys.pop(-1)
            prx,pry,prz = rx,ry,rz

            if sum(diffs) > THRES or sum(rys) > 25 or sum(drys) > 5:
                # print("large motion")
                continue
            frame_count += 1

            frame = imutils.resize(frame, width=400)
            x, y, w, h = boxes
            
            w,h = initbox[2],initbox[3]
            x1=max(0,int(y/2-20))
            x2=min(int((y+h)/2+20),frame.shape[0])
            y1=max(0,int(x/2-20))
            y2=min(int((x+w)/2+20),frame.shape[1])
            out_frame = frame[x1:x2,y1:y2]
            canvas = 225*np.ones((int(h/2)+41,int(h/2)+41,3), np.uint8)
            canvas[max(0,-int(y/2-20)):(max(0,-int(y/2-20))+out_frame.shape[0]),max(0,-int(x/2-20)):(max(0,-int(x/2-20))+out_frame.shape[1])] = out_frame
            pts1 = np.float32([[0,0],[0,canvas.shape[0]],[canvas.shape[0],canvas.shape[0]],[canvas.shape[0],0]])
            pts2 = np.float32([[0,0],[0,250],[250,250],[250,0]])

            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(canvas,M,(250,250))

            # Display the resulting frame
            dst = stabilizer.stabilize_frame(input_frame=dst,smoothing_window=30,border_type='replicate')
            if frame_count < 31:
                continue
            
            dst = dst[13:237,13:237]

            video.write(dst)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    cv2.destroyAllWindows()
    fvs.stop()
