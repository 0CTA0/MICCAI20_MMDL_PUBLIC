import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import pdb

RESNET_FEATURE = 3072

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


###''' self-attention; relation-attention '''

class ResNet_AT(nn.Module):
    def __init__(self, block, layers, num_classes=2, end2end=True):
        self.inplanes = 64
        self.end2end = end2end
        super(ResNet_AT, self).__init__()
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6,1))
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.7)
        self.sigmoid = nn.Sigmoid()

        #self.beta = nn.Sequential(nn.Linear(1024*6, 1),
        self.beta = nn.Sequential(nn.Linear(512*6, 1),
                                  nn.Sigmoid())
        self.pred_fc2 = nn.Sequential(nn.Linear(512*6, 1),
                                  nn.Sigmoid())
        # self.pred_fc2 = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x='', phrase='train', vectors='',index_matrix=''):
        if phrase == 'train':
            f1 = x[:,0:3,:,:]
            f2 = x[:,3:6,:,:]
            f3 = x[:,6:9,:,:]
            f4 = x[:,9:12,:,:]
            f1 = self.conv0(f1)
            f2 = self.conv0(f2)
            f3 = self.conv0(f3)
            f4 = self.conv0(f4)
            f1 = f1-f2
            f2 = f2-f3
            f3 = f3-f4
            f1 = self.conv1(f1)
            f1 = self.bn1(f1)
            f1 = self.relu(f1)
            f1 = self.maxpool(f1)
            f1 = self.layer1(f1)
            f1 = self.layer2(f1)
            f1 = self.layer3(f1)
            f1 = self.layer4(f1)
            f1 = self.avgpool(f1)
            f1 = f1.squeeze(3)  # f[16, 512, 4, 1] ---> f[16, 512,4]
            f1 = f1.view(-1,512*6)
            output1 = self.dropout2(f1)
            prec_feature = self.sigmoid(output1)
            pred_score1 = self.pred_fc2(output1)
            f2 = self.conv1(f2)
            f2 = self.bn1(f2)
            f2 = self.relu(f2)
            f2 = self.maxpool(f2)
            f2 = self.layer1(f2)
            f2 = self.layer2(f2)
            f2 = self.layer3(f2)
            f2 = self.layer4(f2)
            f2 = self.avgpool(f2)
            f2 = f2.squeeze(3)  # f[16, 512, 4, 1] ---> f[16, 512,4]
            f2 = f2.view(-1,512*6)
            output2 = self.dropout2(f2)
            pred_score2 = self.pred_fc2(output2)
            f3 = self.conv1(f3)
            f3 = self.bn1(f3)
            f3 = self.relu(f3)
            f3 = self.maxpool(f3)
            f3 = self.layer1(f3)
            f3 = self.layer2(f3)
            f3 = self.layer3(f3)
            f3 = self.layer4(f3)
            f3 = self.avgpool(f3)
            f3 = f3.squeeze(3)  # f[16, 512, 4, 1] ---> f[16, 512,4]
            f3 = f3.view(-1,512*6)
            output3 = self.dropout2(f3)
            pred_score3 = self.pred_fc2(output3)
            return pred_score1,pred_score2,pred_score3,prec_feature

        if phrase == 'eval':
            f1 = x[:,0:3,:,:]
            f2 = x[:,3:6,:,:]
            f1 = self.conv0(f1)
            f2 = self.conv0(f2)
            f1 = f1-f2
            f1 = self.conv1(f1)
            f1 = self.bn1(f1)
            f1 = self.relu(f1)
            f1 = self.maxpool(f1)
            f1 = self.layer1(f1)
            f1 = self.layer2(f1)
            f1 = self.layer3(f1)
            f1 = self.layer4(f1)
            f1 = self.avgpool(f1)
            f1 = f1.squeeze(3)  # f[16, 512, 4, 1] ---> f[16, 512,4]
            f1 = f1.view(-1,512*6)
            output1 = self.dropout2(f1)
            pred_score1 = self.sigmoid(output1)
            return pred_score1


class LSTMClassifier(nn.Module):
    def __init__(self, args):
        super(LSTMClassifier, self).__init__()  
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        # Co = args.kernel_num
        # Ks = args.kernel_sizes
        hidden_dim = 256
        n_layers = 1
        output_dim = 2
        self.embedding_dim = D
        self.hidden_dim = hidden_dim
        self.vocab_size = V

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, bidirectional=True)

        self.hidden2out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)


    def init_hidden(self, batch_size):
        return(torch.autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).cuda(),
                        torch.autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).cuda())


    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(0))
        
        embeds = self.embedding(batch.to(dtype=torch.int64, device='cuda'))
        packed_input = pack_padded_sequence(embeds, lengths,batch_first=True)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)
        # ht is the last hidden state of the sequences
        output = self.dropout_layer(ht[-1])

        return output

class DummyLayer(nn.Module):
    def forward(self, x):
        return x

def ResNetClassifier(pretrained=False, **kwargs):
    # Constructs base a ResNet-34 model.
    model = ResNet_AT(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


class fusion(nn.Module):
    def __init__(self, args,pretrained=False, **kwargs):
        super(fusion, self).__init__()
        self.lstm = ResNetClassifier(args)
        self.resnet = resnet()
        # self.resnet.pred_fc2 = DummyLayer()
        self.outlayer = nn.Linear(
            in_features=RESNET_FEATURE+self.lstm.hidden_dim, out_features=2)
        self.softmax = nn.Sigmoid()

    def forward(self, image, text, lengths, phrase='train'):
        if phrase=='train':
            p1,p2,p3,out1 = self.resnet(image,phrase)
            out2 = self.lstm(text, lengths)
            # out.requires_grad = True
            # print(out2)
            out = torch.cat((out1, out2), 1)
            out.retain_grad()
            out = self.outlayer(out)
            out = self.softmax(out)
            return p1,p2,p3,out
        if phrase=='eval':
            out1 = self.resnet(image,phrase)
            out2 = self.lstm(text, lengths)
            # print(out2)
            out = torch.cat((out1, out2), 1)
            out = self.outlayer(out)
            out = self.softmax(out)
            return out
