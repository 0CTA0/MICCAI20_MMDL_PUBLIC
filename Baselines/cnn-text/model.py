import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, x_len):
        x = self.embed(x)  # (N, W, D)
        
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        # print(logit.shape)
        return logit

class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        hidden_dim = 16
        n_layers = 1
        output_dim = 1
        self.embedding = nn.Embedding(V, D)
        self.lstm = nn.LSTM(D, hidden_dim, num_layers=n_layers,
                            bidirectional=True)
        self.Ws = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        self.bs = nn.Parameter(torch.zeros((output_dim, )))
        nn.init.uniform_(self.Ws, -0.3, 0.3)
        nn.init.uniform_(self.bs, -0.3, 0.3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, x_len):
        x = self.embedding(x)
        # torch.set_default_tensor_type(torch.FloatTensor)
        # x_len = torch.as_tensor(x_len, dtype=torch.int64, device='cpu')
        # print(x_len.type())
        x = pack_padded_sequence(x, x_len,batch_first=True)
        # torch.set_default_tensor_type(torch.cuda.FloatTensor)
        H, (h_n, c_n) = self.lstm(x)
        h_n = self.dropout(h_n)
        h_n = torch.squeeze(h_n)
        res = torch.matmul(h_n, self.Ws) + self.bs
        y = F.relu(F.softmax(res, dim=1))
        y = y.view(x_len.shape[0],-1)
        # y.size(batch_size, output_dim)
        return y

class LSTMClassifier(nn.Module):
    def __init__(self, args):
        super(LSTMClassifier, self).__init__()  
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        hidden_dim = 512
        n_layers = 1
        output_dim = 2
        self.embedding_dim = D
        self.hidden_dim = hidden_dim
        self.vocab_size = V

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, bidirectional=True)

        self.hidden2out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.5)


    def init_hidden(self, batch_size):
        return(torch.autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).cuda(),
                        torch.autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)).cuda())


    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(0))
        embeds = self.embedding(batch)
        # print(embeds)
        packed_input = pack_padded_sequence(embeds, lengths,batch_first=True)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        # output = self.softmax(output)

        return output