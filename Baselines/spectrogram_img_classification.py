from __future__ import print_function
import argparse
import torch
# print(torch.__version__)
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import time
import math
from PIL import Image
import random
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn as cudnn
import os
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data as data
from imblearn.metrics import sensitivity_specificity_support
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, recall_score
import numpy as np
import torch.nn.functional as F
import cv2
import pdb
import torchvision.models as models
from tqdm import tqdm

weights = [1.0,0.5]
class_weights = torch.FloatTensor(weights).cuda()

cate2label = {'CK+':{0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                     'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 5,'Sad': 4,'Surprise': 6},

              'AFEW':{0: 'Happy',1: 'Angry',2: 'Disgust',3: 'Fear',4: 'Sad',5: 'Neutral',6: 'Surprise',
                  'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Neutral': 5,'Sad': 4,'Surprise': 6},
            'Cust':{'nonstroke':0, 'stroke':1}}
cate2label = cate2label['Cust']


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)  # first position is score; second position is pred.
    pred = pred.t()  # .t() is T of matrix (16 * 1) -> (1 * 16)
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target.view(1,2,2,-1): (256,) -> (1, 2, 2, 64)
    correct_k = correct.view(-1).float().sum(0)
    res = correct_k.mul_(100.0 / batch_size)
    return res, pred


def adjust_learning_rate(optimizer, epoch, learning_rate, end_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in [round(end_epoch * 0.333), round(end_epoch * 0.666)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.2

        learning_rate = learning_rate* 0.2
        print('Adjust_learning_rate ' + str(epoch))
        print('New_LearningRate: {}'.format(learning_rate))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, at_type=''):

    if not os.path.exists('./model'):
        os.makedirs('./model')

    epoch = state['epoch']
    save_dir = './model/'+at_type+'_' + str(epoch) + '_' + str(round(float(state['prec1']), 4))
    torch.save(state, save_dir)
    print(save_dir)

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
        video_list.append("stroke/"+item+" stroke")
    stc = len(video_list)
    for item in nonstrokelist:
        video_list.append("nonstroke/"+item+" nonstroke")
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

        imgs_first.append((video_path, label))
        ###  return video frame index  #####
        video_names.append(video_name[:-4])
        index.append(np.ones(len(video_list)) * cnt)
        cnt = cnt + 1
    index = np.concatenate(index, axis=0)
    return imgs_first, index , video_names, nonstroke_weight

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

def LoadData(root, current_fold, totallist, batchsize_train, batchsize_eval):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = VideoDataset(
        labelid = 1,
        current = current_fold,
        video_root=root,
        video_list=totallist,
        transform=transform)

    val_dataset = VideoDataset(
        labelid = 0,
        current = current_fold,
        video_root=root,
        video_list=totallist,
        transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)


    # return train_loader, val_loader
    return train_loader, val_loader, train_dataset, val_dataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Linear(in_features=802816, out_features=2)


    def forward(self, image):
        out1 = self.resnet(image)
        out1 = out1.view(out1.shape[0], -1)
        out = self.fc(out1)
        return out

TOTAL_PREDS = []
TOTAL_TARGETS = []
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'
parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.7, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=400, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()
# print('The attention is ' + at_type)
    
def main():
    global args, best_prec1, TOTAL_PREDS, TOTAL_TARGETS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('learning rate:', args.lr)
    ''' Load data '''
    arg_fold = 5
    arg_batchsize_train= 32
    arg_batchsize_eval= 32
    arg_root = "spec/"
    # arg_root = "images_labeled/"
    totallist = []
    kf = KFold(n_splits=arg_fold,shuffle=True)
    strokedir = os.listdir(arg_root+"stroke")
    for train1, test1 in kf.split(strokedir):
        #print("%s %s" % (train, test))
        totallist.append([strokedir[x] for x in train1])
        totallist.append([strokedir[x] for x in test1])
    
    nonstrokedir = os.listdir(arg_root+"nonstroke")
    for train1, test1 in kf.split(nonstrokedir):
        #print("%s %s" % (train, test))
        totallist.append([nonstrokedir[x] for x in train1])
        totallist.append([nonstrokedir[x] for x in test1])
    print('args.lr', args.lr)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ans = []
    for i in range(arg_fold):
        best_prec1 = 0
        # _structure = newModel.resnet18_AT()
        # _parameterDir = './model/resnet18-5c106cde.pth'
        # #_parameterDir = './model/self_relation-attention_5_83.5088'
        # model = newload.LoadParameter(_structure, _parameterDir)
        

        criterion = nn.CrossEntropyLoss().cuda()
        
        cudnn.benchmark = True
        # train_loader, val_loader = newload.LoadData(arg_root, i, totallist, arg_batchsize_train, arg_batchsize_eval)
        train_loader, val_loader, train_dataset, val_dataset = LoadData(arg_root, i, totallist, arg_batchsize_train, arg_batchsize_eval)
        net = models.resnet18(pretrained=True)
        NUM_FEATURE = net.fc.in_features
        net.fc = nn.Linear(NUM_FEATURE, 2)
        # net = CNN()
        net = net.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        best_pred = None
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args.lr, args.epochs)
            train(train_loader, net, criterion, optimizer,device, epoch,train_dataset.__len__()/arg_batchsize_train)
            
            prec1, target, pred = val(val_loader, net, criterion, device)
            print("validated")
            is_best = prec1 > best_prec1
            if is_best:
                print('better model!')
                best_prec1 = max(prec1, best_prec1)
                best_pred = pred
            else:
                print('Model too bad & not save')
        ans.append(best_prec1)
        print(target,best_pred)
        TOTAL_TARGETS+=target
        TOTAL_PREDS+=best_pred
    print(sum(ans) / len(ans))
    print("Accuracy: ",str(accuracy_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print("Precision: ",str(precision_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print("Recall: ",str(recall_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print("AUC: ",str(roc_auc_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print(sensitivity_specificity_support(TOTAL_TARGETS,TOTAL_PREDS))

def train(trainloader, net, criterion, optimizer, device, epoch, length):
    print('------------------------[epoch %d]------------------------' %
          (epoch))
    start = time.time()
    running_loss = []
    correct = 0
    total = 0
    for (images, labels, _) in tqdm(trainloader, total=length, unit="it"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        running_loss.append(loss.item())
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()
    end = time.time()
    print('[epoch %d] Train loss: %.3f eplased time %.3f' %
          (epoch, np.asarray(running_loss).mean(), end-start))
    print('Train Accuracy: %d %%' % (
        100 * correct / total))

    
    
def val(valloader, net, criterion, device):
    global TOTAL_PREDS, TOTAL_TARGETS
    all_lab = []
    all_pred = []
    miss_accept = 0
    miss_reject = 0
    correct_accept = 0
    correct_reject = 0
    total_accept = 0
    total_reject = 0
    running_loss = []
    with torch.no_grad():
        for data in valloader:
            images, labels, _ = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)

            total_accept += (labels == 1).sum().item()
            total_reject += (labels == 0).sum().item()
            for i, label in enumerate(labels):
                if (predicted[i] != label & label == 1):
                    miss_reject += 1
                if (predicted[i] != label & label == 0):
                    miss_accept += 1
                if (predicted[i] == label & label == 1):
                    correct_accept += 1
                if (predicted[i] == label & label == 0):
                    correct_reject += 1
            # correct_accept += (predicted == labels & labels == 1).sum().item()
            # correct_reject += (predicted == labels & labels == 0).sum().item()
            all_lab += labels.cpu().tolist()
            all_pred += predicted.cpu().tolist()
    print('Val loss: %.3f' %
          (np.asarray(running_loss).mean()))
    total = miss_accept+correct_accept
    print(miss_accept,miss_reject,correct_accept,correct_reject)
    accur = (correct_accept+correct_reject)/(miss_accept+miss_reject+correct_accept+correct_reject)
    print("Accuracy: ",accur)
    return accur, all_lab,all_pred


if __name__ == '__main__':
    main()
