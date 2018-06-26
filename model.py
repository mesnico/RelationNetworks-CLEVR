import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

import pdb

DEBUG = False
def debug_print(msg):
    if DEBUG:
        print(msg)


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
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)
        return x


class QuestionEmbedModel(nn.Module):
    def __init__(self, in_size, embed=32, hidden=128):
        super(QuestionEmbedModel, self).__init__()
        
        self.wembedding = nn.Embedding(in_size + 1, embed)  #word embeddings have size 32
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)  # Input dim is 32, output dim is the question embedding
        self.hidden = hidden
        
    def forward(self, question):
        #calculate question embeddings
        wembed = self.wembedding(question)
        # wembed = wembed.permute(1,0,2) # in lstm minibatches are in the 2-nd dimension
        self.lstm.flatten_parameters()
        _, hidden = self.lstm(wembed) # initial state is set to zeros by default
        qst_emb = hidden[0] # hidden state of the lstm. qst = (B x 128)
        #qst_emb = qst_emb.permute(1,0,2).contiguous()
        #qst_emb = qst_emb.view(-1, self.hidden*2)
        qst_emb = qst_emb[0]
        
        return qst_emb

class RelationalLayerBase(nn.Module):
    def __init__(self, in_size, out_size, qst_size, hyp):
        super().__init__()
        
        self.on_gpu = False
        self.hyp = hyp
        self.qst_size = qst_size
        self.in_size = in_size
        self.out_size = out_size

    def cuda(self):
        self.on_gpu = True
        super().cuda()
    

class RelationalLayer(RelationalLayerBase):
    def __init__(self, in_size, out_size, qst_size, hyp, extraction=False):
        super().__init__(in_size, out_size, qst_size, hyp)

        self.quest_inject_position = hyp["question_injection_position"]
        self.aggreg_position = hyp["aggregation_position"]
        #dropouts
        self.dropouts = {int(k):nn.Dropout(p=v) for k,v in hyp["dropouts"].items()}
        
        self.in_size = in_size

        #create all g layers
        self.g_layers = []
        self.g_layers_size = hyp["g_layers"]
        for idx,g_layer_size in enumerate(hyp["g_layers"]):
            in_s = in_size if idx==0 else hyp["g_layers"][idx-1]
            out_s = g_layer_size                
            if idx==self.quest_inject_position:
                #create the h layer. Now, for better code organization, it is part of the g layers pool. 
                l = nn.Linear(in_s+qst_size, out_s)
            else:
                #create a standard g layer.
                l = nn.Linear(in_s, out_s)
            self.g_layers.append(l)	
        self.g_layers = nn.ModuleList(self.g_layers)
        self.extraction = extraction
    
    def forward(self, x, qst):
        # x = (B x 8*8 x 24)
        # qst = (B x 128)
        """g"""
        b, d, k = x.size()
        qst_size = qst.size()[1]
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x, 1)                   # (B x 1 x 64 x 26)
        x_i = x_i.repeat(1, d, 1, 1)                    # (B x 64 x 64 x 26)
        x_j = torch.unsqueeze(x, 2)                   # (B x 64 x 1 x 26)
        #x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, d, 1)                    # (B x 64 x 64 x 26)
        
        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)                  # (B x 64 x 64 x 2*26)
        
        # reshape for passing through network
        x_ = x_full.view(b * d**2, self.in_size)

        #create g and inject the question at the position pointed by quest_inject_position.
        for idx, (g_layer, g_layer_size) in enumerate(zip(self.g_layers, self.g_layers_size)):
            in_size = self.in_size if idx==0 else self.g_layers_size[idx-1]

            if idx==self.aggreg_position:
                debug_print('{} - Aggregation'.format(idx))
                x_ = x_.view(b,-1,in_size)
                x_ = x_.sum(1)

            if idx==self.quest_inject_position:
                debug_print('{} - Question injection'.format(idx))
                x_img = x_.view(b,-1,in_size)
                n_couples = x_img.size()[1]

                # add question everywhere
                unsq_qst = torch.unsqueeze(qst, 1)                      # (B x 1 x 128)
                unsq_qst = unsq_qst.repeat(1, n_couples**2, 1)          # (B x 64*64 x 128)
                #unsq_qst = torch.unsqueeze(unsq_qst, 2)                 # (B x 64 x 1 x 128)
                #unsq_qst = unsq_qst.repeat(1,1,n_couples,1)

                # questions inserted
                
                x_concat = torch.cat([x_img,unsq_qst],2) #(B x 64*64 x 128+256)
                x_ = x_concat.view(b*(n_couples**2),in_size+self.qst_size)
                
                x_ = g_layer(x_)

                if self.extraction:
                    return None                
            else:
                x_ = g_layer(x_)

            debug_print('{} - Layer. Output dim: {}'.format(idx, x_.size()))
            
            if idx in self.dropouts:
                debug_print('{} - Dropout p={}'.format(idx, self.dropouts[idx].p))
                x_ = self.dropouts[idx](x_)
                
            #apply ReLU after every layer except the last
            if idx!=len(self.g_layers_size)-1:
                debug_print('{} - ReLU'.format(idx))
                x_ = F.relu(x_)

        if DEBUG:
            pdb.set_trace()        

        return F.log_softmax(x_, dim=1)

class RN(nn.Module):
    def __init__(self, args, hyp, extraction=False):
        super(RN, self).__init__()
        self.coord_tensor = None
        self.on_gpu = False
        
        # CNN
        self.conv = ConvInputModel()
        self.state_desc = hyp['state_description']            
            
        # LSTM
        hidden_size = hyp["lstm_hidden"]
        self.text = QuestionEmbedModel(args.qdict_size, embed=hyp["lstm_word_emb"], hidden=hidden_size)
        
        # RELATIONAL LAYER
        self.rl_in_size = hyp["rl_in_size"]
        self.rl_out_size = args.adict_size
        self.rl = RelationalLayer(self.rl_in_size, self.rl_out_size, hidden_size, hyp, extraction) 
        if hyp["question_injection_position"] != 0:          
            print('Supposing IR model')
        else:     
            print('Supposing original DeepMind model')

    def forward(self, img, qst_idxs):
        if self.state_desc:
            x = img # (B x 12 x 8)
        else:
            x = self.conv(img)  # (B x 24 x 8 x 8)
            b, k, d, _ = x.size()
            x = x.view(b,k,d*d) # (B x 24 x 8*8)
            
            # add coordinates
            if self.coord_tensor is None or torch.cuda.device_count() == 1:
                self.build_coord_tensor(b, d)                  # (B x 2 x 8 x 8)
                self.coord_tensor = self.coord_tensor.view(b,2,d*d) # (B x 2 x 8*8)
            
            x = torch.cat([x, self.coord_tensor], 1)    # (B x 24+2 x 8*8)
            x = x.permute(0, 2, 1)    # (B x 64 x 24+2)
        
        qst = self.text(qst_idxs)
        y = self.rl(x, qst)
        return y
       
    # prepare coord tensor
    def build_coord_tensor(self, b, d):
        coords = torch.linspace(-d/2., d/2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x,y))
        # broadcast to all batches
        # TODO: upgrade pytorch and use broadcasting
        ct = ct.unsqueeze(0).repeat(b, 1, 1, 1)
        self.coord_tensor = Variable(ct, requires_grad=False)
        if self.on_gpu:
            self.coord_tensor = self.coord_tensor.cuda()
    
    def cuda(self):
        self.on_gpu = True
        self.rl.cuda()
        super(RN, self).cuda()
        
