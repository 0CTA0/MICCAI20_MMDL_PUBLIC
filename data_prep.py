import subprocess
import os
import sys
import cv2
import glob
import shutil
import json
import csv
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile
from matplotlib import pyplot as plt
from PIL import Image

def graph_spectrogram(wav_file,name):
    rate, data = wavfile.read(os.curdir+"/RawData/Audio/"+wav_file+".wav")
    # print (type(data), len(data))
    nfft = 512  # Length of the windowing segments
    fs = 10000  # Sampling frequency
    pxx, freqs, bins, im = plt.specgram(data[:,0], Fs=fs)
    plt.axis('off')
    img_dir = os.curdir+'/Feature/Spectrograms/'+name+'/'+wav_file + '.png'
    plt.savefig(img_dir,
                dpi=300,  # Dots per inch
                bbox_inches='tight',
                pad_inches=0)  # Spectrogram saved as a .png
    # try:
    im = Image.open(img_dir)
    rgb_im = im.convert('RGB')
    rgb_im.save(img_dir)
    plt.close()


if not os.path.exists(os.curdir+"/Feature/"):
    os.makedirs(os.curdir+"/Feature/Spectrograms/")
    os.makedirs(os.curdir+"/Feature/Crop/")
    os.makedirs(os.curdir+"/Feature/Frames/")


#Extract wav-> RawData/Video/* ~ RawData/Audio/*
for root, dirs, files in os.walk(os.curdir+"/RawData/Video/"):
    for filename in files:
        command = "ffmpeg -i "+os.curdir+"/RawData/Video/"+filename+" -ab 160k -ac 2 -ar 44100 -vn " + os.curdir+"/RawData/Audio/"+filename[:-4]+".wav"
        # subprocess.call(command, shell=True)

# Video Tracking & Cropping
f = os.listdir(os.curdir+"/RawData/Video/")
for item in f:
    print(item[:-4])
    # video_filepath = line.strip("\n")
    os.chdir(os.curdir+"/faceTracking/")
    os.system("python run.py -v "+item[:-4])
    os.chdir("../")

for name in ['stroke','nonstroke']:
    filelist = []
    with open(os.curdir+"/RawData/"+name+".txt") as f:
        cases = f.read().splitlines()
        filelist = cases
    if not os.path.exists(os.curdir+"/Feature/Spectrograms/"+name):
        os.makedirs(os.curdir+"/Feature/Spectrograms/"+name)

    for item in filelist:
        #Extract frames-> RawData/Video/* ~ Feature/Frames/*(s/n)/*
        vidcap = cv2.VideoCapture(os.curdir+"/Feature/Crop/"+item+".avi")
        success,image = vidcap.read()
        count = 1
        # print(item)
        dirs = item.split(".")[0]
        if not os.path.exists(os.curdir+"/Feature/Frames/"+name+"/"+dirs):
            os.makedirs(os.curdir+"/Feature/Frames/"+name+"/"+dirs)
        while success:
            cv2.imwrite(os.curdir+"/Feature/Frames/"+name+"/"+dirs+"/"+str(count)+".jpg", image)     # save frame as JPEG file
            success,image = vidcap.read()
            count += 1
        #Generate spectrograms-> RawData/Audio/* ~ Feature/Spectrograms/*(s/n)/*
        graph_spectrogram(item,name)
            