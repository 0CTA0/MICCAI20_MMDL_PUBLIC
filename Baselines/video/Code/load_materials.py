from __future__ import print_function
import torch
print(torch.__version__)
import torch.utils.data
import torchvision.transforms as transforms
from Code import stroketrain

cate2label = {'CK+':{0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                     'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Contempt': 5,'Sad': 4,'Surprise': 6},

              'AFEW':{0: 'Happy',1: 'Angry',2: 'Disgust',3: 'Fear',4: 'Sad',5: 'Neutral',6: 'Surprise',
                  'Angry': 1,'Disgust': 2,'Fear': 3,'Happy': 0,'Neutral': 5,'Sad': 4,'Surprise': 6},
            'Cust':{'nonstroke':0, 'stroke':1}}
cate2label = cate2label['Cust']


def LoadData(root, current_fold, totallist, batchsize_train, batchsize_eval):

    train_dataset = stroketrain.VideoDataset(
        labelid = 1,
        current = current_fold,
        video_root=root,
        video_list=totallist,
        transform=transforms.Compose([transforms.ToTensor()]))

    val_dataset = stroketrain.VideoDataset(
        labelid = 0,
        current = current_fold,
        video_root=root,
        video_list=totallist,
        transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize_eval, shuffle=False,
        num_workers=8, pin_memory=True)


    return train_loader, val_loader

def LoadParameter(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model
