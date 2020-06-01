#coding=utf-8
import os, sys, shutil
import random as rd
from PIL import Image
import numpy as np
import random
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import pdb
import re
import csv
from Code import util
try:
    import cPickle as pickle
except:
    import pickle
import nltk
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

cate2label = {'Cust':{'nonstroke':0, 'stroke':1}}
cate2label = cate2label['Cust']

def load_all(labelid, roundid, video_root, video_list,d,word2vec):
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
    rawtext = []
    cnt = 0
    index = []
    video_names = []
    stoplist = stopwords.words('english') + list(string.punctuation)
    stemmer = SnowballStemmer('english')
    
    
    for line in video_list:
        video_label = line.split()

        video_name = video_label[0]  # name of video
        label = cate2label[video_label[1]]  # label of video
        rawtext.append(d[video_name.split('/')[1]][0])
        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
        ###  for sampling triple imgs in the single video_path  ####

        img_lists = os.listdir(video_path)
        img_lists.sort(key=lambda f: int(re.sub('\D', '', f)))  # sort files by ascending
        newlist =[]
        img_count = len(img_lists)  # number of frames in video
        for k in range(0,img_count-4):
            if k==400:
                break
            newlist.append([img_lists[k], img_lists[k+1], img_lists[k+2], img_lists[k+3]])
        img_lists = newlist
        img_count = len(img_lists)
        for frame in img_lists:
            imgs_first.append(([os.path.join(video_path, frame[0]),os.path.join(video_path, frame[1]),os.path.join(video_path, frame[2]),os.path.join(video_path, frame[3])], label))
        ###  return video frame index  #####
        video_names.append(video_name)
        index.append(np.ones(img_count) * cnt)
        cnt = cnt + 1
    index = np.concatenate(index, axis=0)
    text_ = [[word.lower() for word in case.translate(str.maketrans('', '', string.punctuation)).split()] for case in rawtext]
    word2vec = Word2Vec.load("w2v")
    # print(text)
    text__ = [[word2vec.wv.vocab[token].index for token in t if token in word2vec.wv.vocab] for t in text_]
    vocab_size = len(word2vec.wv.vocab)
    max_len = max([len(case) for case in text__])
    text = [[case + [0] * (max_len - len(case))] for case in text__]
    return imgs_first, text, index, video_list, vocab_size


class FusionDataset(data.Dataset):
    def __init__(self, labelid, current, video_root, video_list, text_file, word2vec, transform=None):

        self.label = labelid
        self.roundid = current
        self.imgs_first, self.text, self.index, self.name_list, self.vocab_size = load_all(labelid, current, video_root, video_list, text_file, word2vec)
        #remain to optimize about the parameter length
        self.transform = transform

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]
        img_first = [Image.open(path_first[0]).convert("RGB"),Image.open(path_first[1]).convert("RGB"),Image.open(path_first[2]).convert("RGB"),Image.open(path_first[3]).convert("RGB")]
        if self.transform is not None:
            firstframe = self.transform(img_first[0])
            secondframe = self.transform(img_first[1])
            thirdframe = self.transform(img_first[2])
            fourthframe = self.transform(img_first[3])
            img_first = torch.cat([firstframe, secondframe,thirdframe,fourthframe],dim=0)
        return img_first, np.asarray(self.text[int(self.index[index])]), target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)

    def get_name(self):
        return self.name_list
