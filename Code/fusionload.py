from __future__ import print_function
import torch
import torch.utils.data
import torchvision.transforms as transforms
from Code import fusiontrain

cate2label = {'Cust':{'nonstroke':0, 'stroke':1}}
cate2label = cate2label['Cust']


def LoadData(root, current_fold, totallist, batchsize_train, batchsize_eval,d,word2vec):

    train_dataset = fusiontrain.FusionDataset(
        labelid = 1,
        current = current_fold,
        video_root=root,
        video_list=totallist,
        text_file=d,
        word2vec=word2vec,
        transform=transforms.Compose([transforms.ToTensor()]))

    val_dataset = fusiontrain.FusionDataset(
        labelid = 0,
        current = current_fold,
        video_root=root,
        video_list=totallist,
        text_file=d,
        word2vec=word2vec,
        transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)


    return train_loader, val_loader, train_dataset, val_dataset

def LoadParameter(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    model_state_dict = _structure.state_dict()
    for key in checkpoint:
        #print(key)
        if ((key == 'fc.weight') | (key == 'fc.bias')):
            pass
        else:
            model_state_dict[key.replace('module.', '')] = checkpoint[key]
    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model
