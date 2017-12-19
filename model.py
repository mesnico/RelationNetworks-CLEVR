import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)
        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x


class QuestionEmbedModel(nn.Module):
    def __init__(self, in_size, embed=32, hidden=128):
        super(QuestionEmbedModel, self).__init__()
        
        self.wembedding = nn.Embedding(in_size + 1, embed, padding_idx=0)  #word embeddings have size 32. Indexes 0 output 0 vectors (variable len questions)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)  # Input dim is 32, output dim is the question embedding
        
    def forward(self, question):   
        #calculate question embeddings
        wembed = self.wembedding(question)
        # wembed = wembed.permute(1,0,2) # in lstm minibatches are in the 2-nd dimension
        self.lstm.flatten_parameters()
        _, hidden = self.lstm(wembed) # initial state is set to zeros by default
        qst_emb = hidden[1] # last layer of the lstm. qst = (B x 128)
        # take the hidden state at the last LSTM layer (as of now, there is only one LSTM layer)
        qst_emb = qst_emb[-1]
        
        return qst_emb
        

class RelationalLayerModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(RelationalLayerModel, self).__init__()
        
        self.coord_tensor = None
        
        self.g_fc1 = nn.Linear(in_size, 256)
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_fc3 = nn.Linear(256, out_size)

        self.dropout = nn.Dropout(p=0.5)

                
    def cuda(self):
        self.coord_tensor = self.coord_tensor.cuda()
        super(RelationalLayerModel, self).cuda()
    
        # prepare coord tensor
    def build_coord_tensor(self, b, d):
        a = torch.arange(0, d**2)
        x = (a/d - d/2)/(d/2.)
        y = (a%d - d/2)/(d/2.)
        ct = torch.stack((x,y), dim=1)
        # broadcast to all batches
        # TODO: upgrade pytorch and use broadcasting
        ct = ct.repeat(b, 1, 1)
        self.coord_tensor = Variable(ct, require_grad=False)
    
    def forward(self, x, qst):
        # x = (B x 24 x 8 x 8)
        # qst = (B x 128)
        """g"""
        b, k, d, _ = x.size()
        qst_size = qst.size()[1]

        x_flat = x.view(b, n_channels, d*d).permute(0,2,1) # (B x 64 x 24)
        
        # add coordinates
        if self.coord_tensor is None:
            self.build_coord_tensor(b, d)                  # (B x 64 x 2)
            
        x_flat = torch.cat([x_flat, self.coord_tensor], 2) # (B x 64 x 24+2)
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)                      # (B x 1 x 128)
        qst = qst.repeat(1, d**2, 1)                       # (B x 64 x 128)
        qst = torch.unsqueeze(qst, 2)                      # (B x 64 x 1 x 128)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)                   # (B x 1 x 64 x 26)
        x_i = x_i.repeat(1, d**2, 1, 1)                    # (B x 64 x 64 x 26)
        x_j = torch.unsqueeze(x_flat, 2)                   # (B x 64 x 1 x 26)
        x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, d**2, 1)                    # (B x 64 x 64 x 26+128)
        
        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)                  # (B x 64 x 64 x 2*26+128)
        
        # reshape for passing through network
        x_ = x_full.view(b * d**4, 2*26 + qst_size)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(b, d**4, 256)
        x_g = x_g.sum(1).squeeze(1)
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = self.dropout(x_f)
        x_f = self.f_fc3(x_f)

        return F.log_softmax(x_f)


class RN(nn.Module):
    def __init__(self, args):
        super(RN, self).__init__()
        # CNN
        self.conv = ConvInputModel()
        
        # LSTM
        hidden_size = 128
        self.text = QuestionEmbedModel(args.qdict_size, embed=32, hidden=hidden_size)
        
        # RELATIONAL LAYER
        # (No. of filters per object + coordinate of object)*2 + question vector
        self.rl_in_size = (24 + 2)*2 + hidden_size
        self.rl_out_size = args.adict_size
        self.rl = RelationalLayerModel(self.rl_in_size, self.rl_out_size)

    def forward(self, img, qst_idxs):
        x = self.conv(img)
        qst = self.text(qst_idxs)
        y = self.rl(x, qst)
        return y
    
    def cuda(self):
        self.rl.cuda()
        super(RN, self).cuda()
        
