# MICCAI20_MMDL_PUBLIC
This is the public repository for MICCAI 2020 paper:
```
Mingli Yu, Tongan Cai, Kelvin Wong, Xiaolei Huang, John Volpi, James Z. Wang and Stephen T.C. Wong. 
Toward Rapid Stroke Diagnosis with Multimodal Deep Learning
```
## Introduction
We propose a multimodal deep learning framework to achieve computer-aided stroke presence assessment on patients with suspicion of stroke showing facial paralysis and speech disorders. We emulate the CPSS and FAST process and record facial videos of patients performing specific speech tests in an acute setting. A novel spatiotemporal proposal method is adopted to extract near-frontal, facial-motion-only frame sequences (data_preap.py). A deep neural network is constructed as the video classification module. Transcribed speech information is also utilized by a text classification module and fused with the video module (fusion.py, fusion_model.py). 

## Data & Directory
We expect the input of data to be the following five parts:
* Video files with embedded audio and packed as .MOV format (/RawData/Video/xxxx.MOV)
* Transcript files given by Google Cloud Speech-to-Text (/RawData/Transcript/xxxx.json)
* Two txt files indicating which cases are stroke/nonstroke (/RawData/(non)stroke.txt)
  * Each entry is the corresponding case number xxxx
* Dlib face landmark data (/shape_predictor_68_face_landmarks.dat)
* Pre-trained PyTorch ResNet-34 file (/Model/resnet34-333f7ec4.pth)

The directory is required to be organized as follows:
```
    .
    ├── Checkpoints
    │   └── ...                   # Saved checkpoints  
    │
    ├── Code                      # Model and utils
    │   ├── fusionload.py         
    │   ├── fusionmodel.py           
    │   ├── fusiontrain.py           
    │   └── util.py                    
    │
    ├── RawData                   # Original data
    │   ├── Audio                 # Audio files extracted from Video, {CASENAME}.wav
    │   │   └── ...            
    │   ├── Video                 # Raw .MOV files as {CASENAME}.MOV
    │   │   └── ...            
    │   ├── Transcript            # Transcription with Google Cloud Speech-to-Text as {CASENAME}.json
    │   │   └── ...            
    │   ├── stroke.txt            # Cases diagnozed with stroke. Each entry is {CASENAME}
    │   └── nonstorke.txt         # Cases diagnozed as no stroke. Each entry is {CASENAME}  
    │
    ├── Feature                   # Extracted feature for the system
    │   ├── Crop                  # Facial videos by performing face tracking, {CASENAME}.avi
    │   │   └── ...            
    │   ├── Frames                # Extracted frames of each cropped movies (/(non)stroke/{CASENAME}/*.jpg)
    │   │   └── ...            
    │   ├── Spectrograms          # Extracted spectrograms (/(non)stroke/{CASENAME}.png)
    │   │   └── ...            
    │   └── data.csv              # Automatically extracted text+label 
    │
    ├── faceTracking              # faceTracking module
    │   └── ...                     
    │
    ├── Model
    │   └── resnet34-333f7ec4.pth # Pre-trainied ResNet-34
    └── ...
```
## Requirement
We run our experiments on a laptop with GTX1070 GPU and 16G RAM running Ubuntu 16.04. 
We specify the required environment/packages as follows:
```
    Python==3.7
    CUDA==9.0
    cudnn==7.6.5
    dlib==19.19
    mxnet-cu90
    cupy-cu90
    ffmpeg
    pytorch
    torchtext
    opencv
    filterpy
    imutils
    matplotlib
    numpy
    pillow
    scipy
    pydub
    vidstab (https://pypi.org/project/vidstab/)
    
```
## Usage
First, follow the instruction of pyECO (https://github.com/StrangerZhang/pyECO)
```
cd faceTracking/eco/features
python setup.py build_ext --inplace
```
After that, put all data in the desinated directory and run
```
python data_prep.py
```
The preprocessing module will work out all the needed features from raw inputs. When this is finished, execute
```
python fusion.py
```
So that the training will start with 5 Fold Cross-Validation.

### References
  * pyECO (https://github.com/StrangerZhang/pyECO): We adopted the ECO tracker for face tracking
  * AudioNet (https://github.com/iamvishnuks/AudioNet): The generation of spectrograms takes reference from this repo
  * Emotion-FAN (https://github.com/Open-Debin/Emotion-FAN): We consider the training pipeline of the Emotion recognition work is helpful for our objective
