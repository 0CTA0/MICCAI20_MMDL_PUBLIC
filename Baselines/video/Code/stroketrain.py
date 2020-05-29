#coding=utf-8
import os, sys, shutil
import random as rd
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import pdb
import csv
try:
    import cPickle as pickle
except:
    import pickle

cate2label = {'CK+':{0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                     'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 5,'Sad': 4,'Surprise': 6},

              'AFEW':{0: 'Happy',1: 'Angry',2: 'Disgust',3: 'Fear',4: 'Sad',5: 'Neutral',6: 'Surprise',
                  'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Neutral': 5,'Sad': 4,'Surprise': 6},
            'Cust':{'nonstroke':0, 'stroke':1}}
cate2label = cate2label['Cust']

def load_imgs_total_frame(labelid, roundid, video_root, video_list):
    imgs_first = list()
    strokelist = []
    nonstrokelist = []
    if labelid == 1:
        strokelist = video_list[roundid*2]
        nonstrokelist = video_list[roundid*2+10]
    else:
        strokelist = video_list[roundid*2+1]
        nonstrokelist = video_list[roundid*2+11]
    # print(labelid,strokelist,nonstrokelist)
    video_list = []
    for item in strokelist:
        tmp = os.listdir(video_root+"stroke/"+item)
        if ".DS_Store" in tmp:
            tmp.remove(".DS_Store")
        for name in tmp:
            video_list.append("stroke/"+item+"/"+name+" stroke")
    stc = len(video_list)
    for item in nonstrokelist:
        tmp = os.listdir(video_root+"nonstroke/"+item)
        if ".DS_Store" in tmp:
            tmp.remove(".DS_Store")
        for name in tmp:
            video_list.append("nonstroke/"+item+"/"+name+" nonstroke")
    if labelid == 1:
        print("Stroke Clips: %d"%stc)
        print("Nonstroke Clips: %d"%(len(video_list)-stc))
        random.shuffle(video_list)
    # print(len(video_list))
    nonstroke_weight = stc / (len(video_list)-stc)
    cnt = 0
    index = []
    video_names = []
    for line in video_list:
        video_label = line.split()

        video_name = video_label[0]  # name of video
        label = cate2label[video_label[1]]  # label of video

        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
        ###  for sampling triple imgs in the single video_path  ####

        img_lists = os.listdir(video_path)
        img_lists.sort()  # sort files by ascending
        img_count = len(img_lists)  # number of frames in video

        for frame in img_lists:
            # pdb.set_trace()
            imgs_first.append((os.path.join(video_path, frame), label))
        ###  return video frame index  #####
        video_names.append(video_name)
        index.append(np.ones(img_count) * cnt)
        cnt = cnt + 1
    index = np.concatenate(index, axis=0)
    return imgs_first, index , video_list, nonstroke_weight

def load_imgs(labelid, roundid, video_root, video_list):
    imgs_first = list()
    strokelist = []
    nonstrokelist = []
    if labelid == 1:
        strokelist = video_list[roundid*2]
        nonstrokelist = video_list[roundid*2+10]
    else:
        strokelist = video_list[roundid*2+1]
        nonstrokelist = video_list[roundid*2+11]

    video_list = []
    if ".DS_Store" in strokelist:
            strokelist.remove(".DS_Store")
    for item in strokelist:
        video_list.append("stroke/"+item+" stroke")
    
    if ".DS_Store" in nonstrokelist:
            nonstrokelist.remove(".DS_Store")
    for item in nonstrokelist:
        video_list.append("nonstroke/"+item+" nonstroke")
    
    random.shuffle(video_list)

    cnt = 0
    index = []
    video_names = []
    for line in video_list:
        video_label = line.split()

        video_name = video_label[0]  # name of video
        label = cate2label[video_label[1]]  # label of video

        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
        ###  for sampling triple imgs in the single video_path  ####

        img_lists = os.listdir(video_path)
        img_lists.sort()  # sort files by ascending
        img_count = len(img_lists)  # number of frames in video

        for frame in img_lists:
            # pdb.set_trace()
            imgs_first.append((os.path.join(video_path, frame), label))
        ###  return video frame index  #####
        video_names.append(video_name)
        index.append(np.ones(img_count) * cnt)
        cnt = cnt + 1
    index = np.concatenate(index, axis=0)
    return imgs_first, index

class VideoDataset(data.Dataset):
    def __init__(self, labelid, current, video_root, video_list, transform=None):

        self.label = labelid
        self.roundid = current
        # self.imgs_first, self.index = load_imgs(labelid, current, video_root, video_list)
        self.imgs_first, self.index, self.name_list, self.weight = load_imgs_total_frame(labelid, current, video_root, video_list)
        
        #remain to optimize about the parameter length
        self.transform = transform

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        return img_first, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)

    def get_name(self):
        return self.name_list