import os
import sys

filelist = os.listdir("../SX")
for item in filelist:
    print(item)
    if item==".DS_Store":
        continue
    if len(item.split(".")) == 2:
        dirs = item.split(".")[0]
        if not os.path.exists("../SX/"+dirs):
            os.makedirs("../SX/"+dirs)
        cmd = "ffmpeg -i ../SX/"+item+" ../SX/"+dirs+"/%04d.jpg"
        os.system(cmd)
