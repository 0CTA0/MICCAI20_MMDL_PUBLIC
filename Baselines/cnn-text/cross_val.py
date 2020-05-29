import json
import csv
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
from sklearn.model_selection import KFold
from imblearn.metrics import sensitivity_specificity_support

from torchtext.data import TabularDataset
from torchtext.data import Field
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, recall_score


d = []

strokes = []
nonstrokes = []

with open("../../stroke.txt") as f:
    for vids in f.read().splitlines():
        strokes.append(vids)
    
with open("../../nonstroke.txt") as f:
    for vids in f.read().splitlines():
        nonstrokes.append(vids)
count = 0
st = 0
nst = 0
for f in os.listdir("../Voice/SX/"):
    with open("../Voice/SX/"+f) as j:
        _data = json.load(j)
        total_transcript = ""
        for items in _data["results"]:
            total_transcript += items["alternatives"][0]["transcript"] 
        # x.append(total_transcript)
        count += 1
        if f[:-5] in strokes:
            # writer.writerow([total_transcript, "stroke"])
            # writer.writerow([total_transcript, 1])
            d.append([total_transcript,1])
            st += 1
        if f[:-5] in nonstrokes:
            nst += 1
            d.append([total_transcript,0])
        # print(total_transcript)
        # break

print("Stroke:%d , Nonstroke: %d"%(st,nst))
TOTAL_PREDS = []
TOTAL_TARGETS = []
accs = []

arg_fold = 5
train_lst = []
test_lst = []
kf = KFold(n_splits=arg_fold,shuffle=True)
for train1, test1 in kf.split(d):
    #print("%s %s" % (train, test))
    train_lst.append([d[x] for x in train1])
    test_lst.append([d[x] for x in test1])

for fd in range(arg_fold):
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=5e-6, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=1024, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=70, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # data 
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=512, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5,6,7', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    args = parser.parse_args()


    acc = None
    tn = 0
    vn = 0
    with open('../train.csv', 'w', newline='') as filetrain:
        with open('../val.csv', 'w', newline='') as fileval:
            writertrain = csv.writer(filetrain)
            writerval = csv.writer(fileval)

            writertrain.writerow(["x", "y"])
            writerval.writerow(["x", "y"])
            
            for entry in enumerate(train_lst[fd]):
                # print(entry)
                writertrain.writerow(entry[1])
            for entry in enumerate(test_lst[fd]):
                writerval.writerow(entry[1])

                
        # load data
    # print(vn,tn)
    print("\nFold %d..."%fd)
    # text_field = data.Field(lower=True)
    # label_field = data.Field(sequential=False)
    # print(text_field,label_field)
    # train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
    # train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)
    tokenize = lambda x: x.split()
    text_field = Field(sequential=True, tokenize=tokenize, lower=True)
    label_field = Field(sequential=False, use_vocab=False)

    stk_datafields = [("x", text_field), # we won't be needing the id, so we pass in None as the field
                    ("y", label_field)]
    train_data,val_data = TabularDataset.splits(
            path='../', train='train.csv', validation='val.csv',
            format='csv',
            skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
            fields=stk_datafields)
    text_field.build_vocab(train_data,val_data)
    label_field.build_vocab(train_data,val_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, val_data), 
                                sort_key=lambda x: len(x.x), 
                                sort_within_batch=False,
                                batch_sizes=(args.batch_size, len(val_data)))

    # print(dev_iter.dtype)
    # update args and print
    args.embed_num = len(text_field.vocab)
    args.class_num = len(label_field.vocab) - 1
    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    # print("\nParameters:")
    # for attr, value in sorted(args.__dict__.items()):
    #     print("\t{}={}".format(attr.upper(), value))


    # model
    # cnn = model.CNN_Text(args)
    # cnn = model.LSTM(args)
    cnn = model.LSTMClassifier(args)

    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))

    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()
            

    # train or predict
    if args.predict is not None:
        label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
        print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    elif args.test:
        try:
            train.eval(test_iter, cnn, args) 
        except Exception as e:
            print("\nSorry. The test dataset doesn't  exist.\n")
    else:
        print()
        try:
            # print(train_iter,dev_iter,cnn)
            acc, best_pred, targ = train.train(train_iter, dev_iter, cnn, args)
            accs.append(acc)
            TOTAL_PREDS+=best_pred
            TOTAL_TARGETS+=targ
        except KeyboardInterrupt:
            print('\n' + '-' * 89)
            print('Exiting from training early')
    print(targ,best_pred)
print("Average Accuracy: %.2f"%(sum(accs)/len(accs)))
print("Accuracy: ",str(accuracy_score(TOTAL_TARGETS,TOTAL_PREDS)))
print("Precision: ",str(precision_score(TOTAL_TARGETS,TOTAL_PREDS)))
print("Recall: ",str(recall_score(TOTAL_TARGETS,TOTAL_PREDS)))
print("AUC: ",str(roc_auc_score(TOTAL_TARGETS,TOTAL_PREDS)))
print(sensitivity_specificity_support(TOTAL_TARGETS,TOTAL_PREDS))
