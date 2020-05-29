from __future__ import print_function
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import os
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from Code import newload, util, newModel
from imblearn.metrics import sensitivity_specificity_support
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, recall_score
import numpy as np
TOTAL_PREDS = []
TOTAL_TARGETS = []
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'
parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=400, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()
# print('The attention is ' + at_type)

def main():
    global args, best_prec1, TOTAL_PREDS, TOTAL_TARGETS

    print('learning rate:', args.lr)
    ''' Load data '''
    arg_fold = 5
    arg_batchsize_train= 64
    arg_batchsize_eval= 32
    arg_root = "Data2/"
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

    ans = []
    for i in range(arg_fold):
        best_prec1 = 0
        _structure = newModel.resnet18_AT()
        _parameterDir = './model/resnet34-333f7ec4.pth'
        #_parameterDir = './model/self_relation-attention_5_83.5088'
        model = newload.LoadParameter(_structure, _parameterDir)


        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        cudnn.benchmark = True
        # train_loader, val_loader = newload.LoadData(arg_root, i, totallist, arg_batchsize_train, arg_batchsize_eval)
        train_loader, val_loader, train_dataset, val_dataset = newload.LoadData(arg_root, i, totallist, arg_batchsize_train, arg_batchsize_eval)
        for epoch in range(args.epochs):
            util.adjust_learning_rate(optimizer, epoch, args.lr, args.epochs)

            train(train_loader, model, criterion, optimizer, epoch)
            prec1 = validate(val_loader, model,val_dataset)

            is_best = prec1 > best_prec1
            if is_best:
                print('better model!')
                best_prec1 = max(prec1, best_prec1)
                util.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'prec1': prec1,
                })
            else:
                print('Model too bad & not save')
        ans.append(best_prec1)
    print(sum(ans) / len(ans))
    print("Accuracy: ",str(accuracy_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print("Precision: ",str(precision_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print("Recall: ",str(recall_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print("AUC: ",str(roc_auc_score(TOTAL_TARGETS,TOTAL_PREDS)))
    print(sensitivity_specificity_support(TOTAL_TARGETS,TOTAL_PREDS))


def train(train_loader, model, criterion, optimizer, epoch):
    global record_
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    losses = util.AverageMeter()
    topframe = util.AverageMeter()
    topVideo = util.AverageMeter()

    # switch to train mode
    output_store_fc = []
    target_store = []
    index_vector = []

    model.train()
    end = time.time()
    for i, (input_first, target_first, index) in enumerate(train_loader):
        target = target_first.cuda()
        input_var = torch.autograd.Variable(input_first) #input: [16,3, 128,128] target:[16]

        target_var = torch.autograd.Variable(target)
        # compute output
        ''' model & full_model'''
        #print(target_var.cpu().numpy().shape())
        pred_score = model(input_var)

        loss = criterion(pred_score,target_var)
        loss = loss.sum()
        #
        output_store_fc.append(pred_score)#16,2
        target_store.append(target)#16,1
        index_vector.append(index)#16,1
        # measure accuracy and record loss
        prec1,_ = util.accuracy(pred_score.data, target)
        losses.update(loss.item(), input_var.size(0))
        topframe.update(prec1, input_var.size(0))

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
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {topframe.val:.3f} ({topframe.avg:.3f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, topframe=topframe))
    index_vector = torch.cat(index_vector, dim=0)  # [16] ... [16]  --->  [97056]
    index_matrix = []
    for i in range(int(max(index_vector)) + 1):
        index_matrix.append(index_vector == i)
    index_matrix = torch.stack(index_matrix, dim=0).cuda().float()  # list to array  --->  [52,97000]
    output_store_fc = torch.cat(output_store_fc, dim=0)  # list to array  --->  [97000, 2]
    target_store = torch.cat(target_store, dim=0).float()  # [97000]
    pred_matrix_fc = index_matrix.mm(output_store_fc)  # [52,97000] * [97000, 2] = [52,2]
    target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
        index_matrix.sum(1)).long()
    prec_video,_ = util.accuracy(pred_matrix_fc.cpu(), target_vector.cpu())
    topVideo.update(prec_video, int(max(index_vector))+ 1)
    print(' *Prec@Video {topVideo.avg:.3f}   *Prec@Frame {topframe.avg:.3f} '.format(topVideo=topVideo, topframe=topframe))
    


def validate(val_loader, model,val_dataset):
    global record_, TOTAL_PREDS,TOTAL_TARGETS
    batch_time = util.AverageMeter()
    topVideo = util.AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    output_store_fc = []
    output_alpha    = []
    target_store = []
    index_vector = []
    with torch.no_grad():
        for i, (input_var, target, index) in enumerate(val_loader):
            # compute output
            target = target.cuda()
            input_var = torch.autograd.Variable(input_var)
            ''' model & full_model'''
            #f, alphas = model(input_var, phrase = 'eval')
            f = model(input_var)
            pred_score = 0
            output_store_fc.append(f)
            # output_alpha.append(alphas)
            target_store.append(target)
            index_vector.append(index)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        # print(index_vector)
        index_vector = torch.cat(index_vector, dim=0)
        index_matrix = []
        for i in range(int(max(index_vector)) + 1):
            index_matrix.append(index_vector == i)
        index_matrix = torch.stack(index_matrix, dim=0).cuda().float() #torch.Size([14, 26750])
        output_store_fc = torch.cat(output_store_fc, dim=0)  #torch.Size([26750, 2048])
        
        # output_alpha = torch.cat(output_alpha, dim=0) #torch.Size([26750, 1])
        target_store = torch.cat(target_store, dim=0).float() #torch.Size([26750])
        # print(output_alpha,output_store_fc)
        #  keywords: mean_fc ; weight_sourcefc; sum_alpha; weightmean_sourcefc 
        # weight_sourcefc = output_store_fc.mul(output_alpha) # torch.Size([26750, 2048])
        # sum_alpha = index_matrix.mm(output_alpha) #torch.Size([14, 1])
        # weightmean_sourcefc = index_matrix.mm(weight_sourcefc).div(sum_alpha) #torch.Size([14, 2048])
        # print(index_matrix)
        target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
            index_matrix.sum(1)).long()  #torch.Size([14])
        # pred_score  = model(vectors=output_store_fc, vm=weightmean_sourcefc, alphas_from1=output_alpha, index_matrix=index_matrix, phrase='eval', AT_level='second_level')
        ''''''
        pred_score = index_matrix.mm(output_store_fc)
        # print(pred_score.cpu().numpy(), target_vector.cpu().numpy(),index_vector)
        prec_video, preds = util.accuracy(pred_score.cpu(), target_vector.cpu())
        TOTAL_PREDS += preds.tolist()[0]
        TOTAL_TARGETS += target_vector.cpu().tolist()
        result_dict = {}
        # print(preds)
        # print(val_dataset.get_name())
        for i,line in enumerate(val_dataset.get_name()):
            sl = line[10:].find('/')+10
            if not line[:sl] in result_dict:
                result_dict[line[:sl]] = 0
            result_dict[line[:sl]] += preds.numpy()[0][i]
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
        print(' *Prec@Case {acc:.3f} '.format(acc=acc))

        return topVideo.avg

if __name__ == '__main__':
    main()
