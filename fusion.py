from __future__ import print_function
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import os
import json
import csv
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from Code import fusionload, util, fusionmodel
from sklearn.model_selection import KFold
import numpy as np
import string
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, recall_score


device="cuda"
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'
parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=400, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()
# print('The attention is ' + at_type)

def main():
    global args, best_prec1
    d = {}
    
    strokes = []
    nonstrokes = []

    with open("../stroke.txt") as f:
        for vids in f.read().splitlines():
            strokes.append(vids)
        
    with open("../nonstroke.txt") as f:
        for vids in f.read().splitlines():
            nonstrokes.append(vids)
    count = 0
    st = 0
    nst = 0
    for f in os.listdir("../Audio/Voice/SX/"):
        with open("../Audio/Voice/SX/"+f) as j:
            _data = json.load(j)
            total_transcript = ""
            for items in _data["results"]:
                total_transcript += items["alternatives"][0]["transcript"] 
            # x.append(total_transcript)
            count += 1
            if f[:-5] in strokes:
                # writer.writerow([total_transcript, "stroke"])
                # writer.writerow([total_transcript, 1])
                d[f[:-5]]=[total_transcript,1]
                st += 1
            if f[:-5] in nonstrokes:
                nst += 1
                d[f[:-5]]=[total_transcript,0]                
            # print(total_transcript)
            # break
    print('learning rate:', args.lr)
    ''' Load data '''
    arg_fold = 5
    arg_batchsize_train= 16
    arg_batchsize_eval= 16
    arg_root = "Data1/"
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
    # print(d)
    allwords=[]
    for key in d:
        words = d[key][0].translate(str.maketrans('', '', string.punctuation)).split()
        # print(d[key][0].translate(str.maketrans('', '', string.punctuation)).split())
        allwords.append(words)
    # print(allwords)
    word2vec = Word2Vec(allwords, min_count=1)
    word2vec.save("w2v")
    ans = []

    TOTAL_PREDS = []
    TOTAL_TARGETS = []
    for i in range(arg_fold):
        best_prec1 = 0
        train_loader, val_loader, train_dataset, val_dataset = fusionload.LoadData(arg_root, i, totallist, arg_batchsize_train, arg_batchsize_eval,d,word2vec)
        args.embed_num = train_dataset.vocab_size
        args.embed_dim = 512
        args.class_num = 1
        model = fusionmodel.fusion(args)
        _structure = fusionmodel.resnet18_AT()
        _parameterDir = './model/resnet34-333f7ec4.pth'
        #_parameterDir = './model/self_relation-attention_5_83.5088'
        model.resnet = fusionload.LoadParameter(_structure, _parameterDir)
        #_parameterDir = './model/self_relation-attention_5_83.5088'
        wi = torch.ones(2)
        # wi[0] = 1
        # wi[1] = 0.45
        model.cuda()
        criterion = nn.CrossEntropyLoss(weight=wi).cuda()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        cudnn.benchmark = True
        trainres=[]
        
        best_pred = None
        for epoch in range(args.epochs):
            tmpres,ss = train(train_loader, model, criterion, optimizer, epoch)
            prec1,predtmp,sss,targ = validate(val_loader, model, val_dataset)
            # print(predtmp.shape,targ.shape)
            util.adjust_learning_rate(optimizer, epoch, args.lr, args.epochs)
            is_best = prec1 > best_prec1
            if is_best:
                best_pred = predtmp
                print(targ.data.tolist(),best_pred.data.tolist()[0])
                print('better model!')
                best_prec1 = max(prec1, best_prec1)
            else:
                print('Model too bad & not save')
        TOTAL_PREDS+=best_pred.data.tolist()[0]
        TOTAL_TARGETS+=targ.data.tolist()
        # print(TOTAL_TARGETS,TOTAL_PREDS)
        ans.append(best_prec1)
    print(sum(ans) / len(ans))
    # print("Average Accuracy: %.2f"%(sum(accs)/len(accs)))
    print("Accuracy: ",str(accuracy_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print("Precision: ",str(precision_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print("Recall: ",str(recall_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print("AUC: ",str(roc_auc_score(TOTAL_TARGETS,TOTAL_PREDS)))


def train(train_loader, model, criterion, optimizer, epoch):
    global record_
    batch_time = util.AverageMeter()
    losses = util.AverageMeter()
    topframe = util.AverageMeter()
    topVideo = util.AverageMeter()

    # switch to train mode
    output_store_fc = []
    target_store = []
    index_vector = []

    model.train()
    end = time.time()
    for i, (input_first, text_feature,target_first, index) in enumerate(train_loader):
        text_feature = text_feature.squeeze(1)
        len_x = torch.as_tensor([i.shape[0] for i in text_feature], dtype=torch.int64)
        target = target_first.cuda()
        imgvar = torch.autograd.Variable(input_first) #input: [16,3, 128,128] target:[16]
        text_feature = text_feature.cuda()
        textvar = torch.autograd.Variable(text_feature)
        target_var = torch.autograd.Variable(target)
        # compute output
        ''' model & full_model'''
        #print(target_var.cpu().numpy().shape())
        # print(np.sum(imgvar[0].cpu().detach().numpy()-imgvar[1].cpu().detach().numpy()))
        p1,p2,p3,pred_score = model(imgvar,textvar,len_x)
        tmploss = (p2-p1).abs().sum()+(p3-p2).abs().sum()
        # print(pred_score)
        # another_pred = 1.0-pred_score.clone()
        # pred_score = torch.cat([pred_score,another_pred],dim = 1)
        # print(pred_score)
        # loss = criterion(pred_score[1:21],target_var[1:21])
        loss0 = criterion(pred_score[1:23],target_var[1:23])
        loss0 = loss0.sum()
        loss = (loss0 - tmploss).abs()
        if epoch >3 and torch.equal(pred_score,pred22):
            loss = loss +0.2
        if epoch>3 and torch.equal(pred22,pred33):
            loss = loss +0.2

        # pred_score = torch.cat([pred_score,another_pred],dim = 1)
        #print(pred_score)
        #print(pred_score.shape,target.shape,index.shape)  #[32,2] [32] [32]
        output_store_fc.append(pred_score)#16,2
        target_store.append(target)#16,1
        index_vector.append(index)#16,1
        # measure accuracy and record loss
        #tmptmp = pred_score[:,0].clone() +3.0
        #pred_score[:,0] = tmptmp
        #print(pred_score)
        #print(target)
        prec1,_ = util.accuracy(pred_score.data, target)
        losses.update(loss.item(), imgvar.size(0))
        topframe.update(prec1, imgvar.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {topframe.val:.3f} ({topframe.avg:.3f})\t'
                .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses, topframe=topframe))
    index_vector = torch.cat(index_vector, dim=0)  # [42624]
    index_matrix = []
    for i in range(int(max(index_vector)) + 1):
        index_matrix.append(index_vector == i)
    index_matrix = torch.stack(index_matrix, dim=0).cuda().float()  # list to array  --->  [66,42624]
    output_store_fc = torch.cat(output_store_fc, dim=0)  # list to array  --->  [42624, 2]
    target_store = torch.cat(target_store, dim=0).float()  # [97000]
    pred_matrix_fc = index_matrix.mm(output_store_fc)  # [52,97000] * [97000, 2] = [52,2]
    target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
        index_matrix.sum(1)).long()
    # print(pred_matrix_fc.cpu(),target_vector.cpu()) #[66,2] [66]
    prec_video,pred_res = util.accuracy(pred_matrix_fc.cpu(), target_vector.cpu())
    # print(prec_video,target_vector.cpu())
    topVideo.update(prec_video, int(max(index_vector))+ 1)
    print(' *Prec@Video {topVideo.avg:.3f}   *Prec@Frame {topframe.avg:.3f} '.format(topVideo=topVideo, topframe=topframe))
    # print(pred_res)
    #print(pred_matrix_fc.cpu())
    return pred_res,pred_matrix_fc.cpu()[:,0]


def validate(val_loader, model, val_dataset):
    global record_
    batch_time = util.AverageMeter()
    topVideo = util.AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    output_store_fc = []
    target_store = []
    index_vector = []
    with torch.no_grad():
        for i, (input_first, text_feature,target_first, index) in enumerate(val_loader):
            text_feature = text_feature.squeeze(1)
            
            # compute output
            len_x = torch.as_tensor([i.shape[0] for i in text_feature], dtype=torch.int64)
            target = target_first.cuda()
            text_feature = text_feature.cuda()

            imgvar = torch.autograd.Variable(input_first) #input: [16,3, 128,128] target:[16]
            textvar = torch.autograd.Variable(text_feature)
            target_var = torch.autograd.Variable(target)
            # compute output
            ''' model & full_model'''
            #print(target_var.cpu().numpy().shape())
            pred_score = model(imgvar,textvar,len_x,phrase = 'eval')
            # another_pred = 1.0-pred_score.clone()
            # pred_score = torch.cat([pred_score,another_pred],dim = 1)
            #print(pred_score)
            # loss = criterion(pred_score[1:21],target_var[1:21])

            # pred_score = model(input_var, )
            # another_pred = 1.0-pred_score.clone()
            # pred_score = torch.cat([pred_score,another_pred],dim = 1)
            output_store_fc.append(pred_score)
            target_store.append(target)
            index_vector.append(index)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        # print(index_vector)

        index_vector = torch.cat(index_vector, dim=0)  # [42624]
        index_matrix = []
        for i in range(int(max(index_vector)) + 1):
            index_matrix.append(index_vector == i)
        index_matrix = torch.stack(index_matrix, dim=0).cuda().float()  # list to array  --->  [66,42624]
        output_store_fc = torch.cat(output_store_fc, dim=0)  # list to array  --->  [42624, 2]
        target_store = torch.cat(target_store, dim=0).float()  # [97000]
        pred_matrix_fc = index_matrix.mm(output_store_fc)  # [52,97000] * [97000, 2] = [52,2]
        target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
            index_matrix.sum(1)).long()
        
        # print(pred_matrix_fc.cpu(),target_vector.cpu()) #[66,2] [66]
        prec_video,preds = util.accuracy(pred_matrix_fc.cpu(), target_vector.cpu())
        result_dict = {}
        for i,line in enumerate(val_dataset.get_name()):
            # print(line)
            s1 = line.find('/')
            s2 = line[s1:].find(' ') + s1
            if not line[s1+1:s2] in result_dict:
                result_dict[line[s1+1:s2]] = 0
            result_dict[line[s1+1:s2]] += preds.numpy()[0][i]
        print(result_dict)
        count = 0
        for key, value in result_dict.items():
            if key[0] =='n':
                if value == 0:
                    count += 1
            else:
                if value != 0:
                    count += 1
        acc = count / len(result_dict) * 100
        topVideo.update(prec_video, int(max(index_vector)) + 1)
        print(' *Prec@Video {topVideo.avg:.3f} '.format(topVideo=topVideo))
        # print(preds)
        # print(prec_video.cpu())
        return topVideo.avg,preds,pred_score.cpu()[:,0],target_vector.cpu()

if __name__ == '__main__':
    main()
