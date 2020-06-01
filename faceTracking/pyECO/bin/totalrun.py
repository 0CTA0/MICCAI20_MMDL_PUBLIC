import os
import sys

filelist = os.listdir("../SX")
for item in filelist:
    print(item)
    if item==".DS_Store":
        continue
    if len(item.split(".")) == 1:
        cmd = "/Users/abc/anaconda3/bin/python3 bin/nowtesttest.py --v ../SX/"+item
        os.system(cmd)
