import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F

weights = [1.0,0.5]
class_weights = torch.FloatTensor(weights).cuda()

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=5e-4)

    steps = 0
    best_acc = 0
    last_step = 0
    best_pred = []
    targ = []
    for epoch in range(1, args.epochs+1):
        model.train()

        for batch in train_iter:
            feature, target = batch.x, batch.y
            # print(feature)
            feature.t_(), target.sub_(0)  # batch first, index align

            len_x = torch.as_tensor([i.shape[0] for i in feature], dtype=torch.int64, device='cpu')
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            # logit = model(feature)
            logit = model(feature,len_x)
            # print(logit)
            loss = F.cross_entropy(logit, target,weight=class_weights)
            # loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc, preds, tar = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    best_pred = preds
                    targ = tar
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
    return best_acc, best_pred,targ

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    with torch.no_grad():
        for batch in data_iter:
            feature, target = batch.x, batch.y
            feature.t_(), target.sub_(0)  # batch first, index align
            
            len_x = torch.as_tensor([i.shape[0] for i in feature], dtype=torch.int64, device='cpu')
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            # logit = model(feature)
            logit = model(feature,len_x)
            loss = F.cross_entropy(logit, target, size_average=False)

            avg_loss += loss.data.item()
            print()
            corrects += (torch.max(logit, 1)
                        [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy, torch.max(logit, 1)[1].view(target.size()).data.tolist(), target.data.tolist()


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    len_x = torch.as_tensor([i.shape[0] for i in x], dtype=torch.int64, device='cpu')
    if cuda_flag:
        x = x.cuda()
    # print(x)
    # output = model(x)
    output = model(x,len_x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
