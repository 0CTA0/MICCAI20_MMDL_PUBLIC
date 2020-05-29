import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pdb
import torchvision.models as models

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((4,1))
        self.dropout = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.alpha = nn.Sequential(nn.Linear(512*4, 1),
                                   nn.Sigmoid())

        self.beta = nn.Sequential(nn.Linear(1024*4, 1),
                                  nn.Sigmoid())
        self.pred_fc1 = nn.Linear(512, 2)
        self.pred_fc2 = nn.Linear(4*1024, 2)

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

    def forward(self, x='', phrase='train', AT_level='first_level',vectors='',vm='',alphas_from1='',index_matrix=''):

        vs = []
        alphas = []

        assert phrase == 'train' or phrase == 'eval'
        assert AT_level == 'first_level' or AT_level == 'second_level' or AT_level == 'pred'
        if phrase == 'train':
            f = x
            #print(f.cpu().numpy().shape)
            f = self.conv1(f)
            f = self.bn1(f)
            f = self.relu(f)
            f = self.maxpool(f)

            f = self.layer1(f)
            f = self.layer2(f)
            f = self.layer3(f)
            f = self.layer4(f)
            f = self.avgpool(f)
            #print(f.cpu().detach().numpy().shape)
            f = f.squeeze(3)  # f[16, 512, 4, 1] ---> f[16, 512,4]
            #for q in range(1,512):
            #    f[1, 512-q] = f[1, 512-q] - f[1, 511-q]
            #f[1,0] = 0

            # MN_MODEL(first Level)
            f = f.view(16,512*4)
            vs = f
            alphas = self.alpha(self.dropout(f))
            #print(alphas.cpu().detach().numpy().shape)
            #vs.append(f)
            #alphas.append(self.alpha(self.dropout(f)))
            vm1 = vs.mul(alphas).div(alphas)
            vs = torch.cat([vs, vm1], dim=1)
            betas = self.beta(self.dropout(vs))
            output = vs.mul(betas * alphas).div(betas * alphas)

            output = self.dropout2(output)
            pred_score = self.pred_fc2(output)

            return pred_score

        if phrase == 'eval':
            if AT_level == 'first_level':
                f = self.conv1(x)#x: 32,3 ,128,128
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)

                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)
                f = f.squeeze(3)# 32,512,4
                # MN_MODEL(first Level)
                f = f.view(-1,512*4)
                alphas = self.alpha(self.dropout(f))

                return f, alphas

            if AT_level == 'second_level':
                vms = index_matrix.permute(1, 0).mm(vm)  # [52, 97000] -> [97000,52] * [381,512] --> [21783, 512]
                print("vm:",vm.shape)
                print(index_matrix.shape)
                print(vms.shape)
                vs_cate = torch.cat([vectors, vms], dim=1)
                print(vs_cate.shape)

                betas = self.beta(self.dropout(vs_cate))
                print(betas.shape)
                ''' keywords: mean_fc ; weight_sourcefc; sum_alpha; weightmean_sourcefc '''
                ''' alpha * beta '''
                weight_catefc = vs_cate.mul(alphas_from1)  # [21570,512] * [21570,1] --->[21570,512]
                print(weight_catefc.shape)
                alpha_beta = alphas_from1.mul(betas)
                print(alpha_beta.shape)
                sum_alphabetas = index_matrix.mm(alpha_beta)  # [380,21570] * [21570,1] -> [380,1]
                print(sum_alphabetas.shape)
                weightmean_catefc = index_matrix.mm(weight_catefc).div(sum_alphabetas)
                print(weightmean_catefc.shape)

                weightmean_catefc = self.dropout2(weightmean_catefc)
                pred_score = self.pred_fc2(weightmean_catefc)
                print(pred_score.shape)

                return pred_score

''' self-attention; relation-attention '''
def resnet18_AT(pretrained=False, **kwargs):
    # Constructs base a ResNet-18 model.
    # model = ResNet_AT(BasicBlock, [2, 2, 2, 2], **kwargs)
    model = models.resnet34()
    # model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    return model
