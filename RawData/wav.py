import subprocess
import os

for root, dirs, files in os.walk(os.curdir+"/Data"):
    for filename in files:
        command = "ffmpeg -i "+os.curdir+"/Data/"+filename+" -ab 160k -ac 2 -ar 44100 -vn " + os.curdir+"/Audio/Audio/"+filename[:-4]+".wav"
        subprocess.call(command, shell=True)