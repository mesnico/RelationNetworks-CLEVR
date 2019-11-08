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
    def __init__(self, in_size, embed=32, hidden=128, bidirectional=False):
        super(QuestionEmbedModel, self).__init__()
        
        self.wembedding = nn.Embedding(in_size + 1, embed, padding_idx=0)  #word embeddings have size 32
        self.lstm = nn.LSTM(embed, hidden, batch_first=True, bidirectional=True)  # Input dim is 32, output dim is the question embedding
        self.hidden = hidden
        self.bidirectional = bidirectional
        
    def forward(self, question, lengths):
        #calculate question embeddings
        wembed = self.wembedding(question)
        # wembed = wembed.permute(1,0,2) # in lstm minibatches are in the 2-nd dimension
        pack_wembed = torch.nn.utils.rnn.pack_padded_sequence(wembed, lengths, batch_first=True)
        self.lstm.flatten_parameters()
        _, hidden = self.lstm(pack_wembed) # initial state is set to zeros by default
        qst_emb = hidden[0] # hidden state of the lstm. qst = (B x 128)

        if (self.bidirectional):
        #bidirectional LSTM
        	qst_emb = qst_emb.permute(1,0,2).contiguous()
        	qst_emb = qst_emb.view(-1, self.hidden*2)
        else:
        	qst_emb = qst_emb[0]
        
        return qst_emb


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        # return x


class QuestionEmbedTransformerModel(nn.Module):
    def __init__(self, in_size, embed=128, num_encoder_layers=6, max_q_len=100):
        super().__init__()
        self.wembedding = nn.Embedding(in_size + 1, embed, padding_idx=0)  # word embeddings have size 'embed'
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed, nhead=8, dim_feedforward=256,
                                                       dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_encoder_layers)
        self.pos_encoder = PositionalEncoding(embed, dropout=0.5, max_len=max_q_len)

    def forward(self, question, lengths):
        # calculate the mask for every question in the batch
        masks = torch.ones(question.shape[0], question.shape[1]).bool().to(question.device)
        for m, l in zip(masks, lengths):
            m[:l] = 0
        wembed = self.wembedding(question)

        wembed = wembed.permute(1, 0, 2)
        wembed = self.pos_encoder(wembed)

        # pass through the transformer
        out_wembed = self.transformer_encoder(wembed, src_key_padding_mask=masks)
        out_wembed = out_wembed.permute(1, 0, 2)  # (B x seq_len x embed)

        # first, manually zero the padding words
        for q, l in zip(out_wembed, lengths):
            q[l:] = 0

        # then, aggregate the word embeddings to obtain the sentence embedding (average over the seq_len)
        qembed = out_wembed.sum(1)     # (B x embed)
        qembed *= torch.rsqrt(lengths.unsqueeze(1).float())
        return qembed


class RelationalBase(nn.Module):
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
    

class RelationalModule(RelationalBase):
    def __init__(self, in_size, out_size, qst_size, hyp, extraction=False):
        super().__init__(in_size, out_size, qst_size, hyp)

        self.quest_inject_position = hyp["question_injection_position"]
        self.aggreg_position = hyp["aggregation_position"]
        #dropouts
        self.dropouts = {int(k):nn.Dropout(p=v) for k,v in hyp["dropouts"].items()}
        
        self.in_size = in_size

        #create aggregation weights
        if 'weighted' in hyp['aggregation']:
            self.aggreg_weights = nn.Parameter(torch.Tensor(12**2 if hyp['state_description'] else 64**2))
            nn.init.uniform_(self.aggreg_weights,-1,1)

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
        self.g_layers.append(nn.Linear(self.g_layers_size[-1], out_size))	
        self.g_layers_size.append(out_size)
        self.g_layers = nn.ModuleList(self.g_layers)
        self.extraction = extraction
        self.aggregation = hyp['aggregation']
       
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
            out_size = g_layer_size if idx!=len(self.g_layers)-1 else self.out_size
            if idx==self.aggreg_position:
                debug_print('{} - Aggregation'.format(idx))
                x_ = x_.view(b,-1,in_size)
                if self.aggregation == 'sum':
                    x_ = x_.sum(1)
                elif self.aggregation == 'avg':
                    x_ = x_.mean(1)
                elif self.aggregation == 'weighted_sum':
                    x_ = torch.matmul(x_.permute(0,2,1), self.aggreg_weights)
                else:
                    raise ValueError('Aggregation not recognized: {}'.format(self.aggregation))
                    
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


class RelationalTransformerModule(RelationalBase):
    def __init__(self, in_size, out_size, qst_size, hyp, num_encoder_layers=6, extraction=False):
        super().__init__(in_size, out_size, qst_size, hyp)
        transformer_layer = nn.TransformerEncoderLayer(d_model=in_size+qst_size, nhead=7, dim_feedforward=256,
                                                      dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_encoder_layers)

        self.f_fc1 = nn.Linear(qst_size + in_size, hyp["f_fc1"])
        self.f_fc2 = nn.Linear(hyp["f_fc1"], hyp["f_fc2"])
        self.f_fc3 = nn.Linear(hyp["f_fc2"], out_size)

    def forward(self, x, qst):
        # x = (B x 8*8 x 24)
        # qst = (B x 128)
        """g"""
        b, d, k = x.size()
        qst_size = qst.size()[1]

        # add question everywhere
        qst = torch.unsqueeze(qst, 1)  # (B x 1 x 128)
        qst = qst.repeat(1, d, 1)  # (B x 64 x 128)

        # cast all pairs against each other
        x_cat = torch.cat([x, qst], 2)

        # pass through the transformer
        x_cat = x_cat.permute(1, 0, 2)     # (64 x B x qst_size+in_size)
        out = self.transformer_encoder(x_cat)
        out = out.permute(1, 0, 2)  # (B x 64 x qst_size+in_size)

        # aggregate
        out = out.sum(1)

        # final processing
        x_f = self.f_fc1(out)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)

        return F.log_softmax(x_f, dim=1)


class RN(nn.Module):
    def __init__(self, args, hyp, extraction=False):
        super(RN, self).__init__()
        self.coord_tensor = None
        self.on_gpu = False
        
        # CNN
        self.conv = ConvInputModel()
        self.state_desc = hyp['state_description']
        
        # RELATIONAL LAYER
        self.rl_in_size = hyp["rl_in_size"]
        self.rl_out_size = args.adict_size
        if 'reasoning_module' not in hyp or hyp['reasoning_module'] == 'rn':
            # LSTM
            hidden_size = hyp["q_embed_dim"]
            bidirectional = hyp["bidirectional"]
            self.text = QuestionEmbedModel(args.qdict_size, embed=hyp["w_embed_dim"], hidden=hidden_size,
                                           bidirectional=bidirectional)
            # RN
            self.rl = RelationalModule(self.rl_in_size, self.rl_out_size, hidden_size * 2 if bidirectional else hidden_size, hyp, extraction)
            print('Using standard RN module')
        elif hyp['reasoning_module'] == 'transformer':
            assert hyp["w_embed_dim"] == hyp["q_embed_dim"]
            self.text = QuestionEmbedTransformerModel(args.qdict_size, embed=hyp["w_embed_dim"])    # w_embed_dim == q_embed_dim
            self.rl = RelationalTransformerModule(self.rl_in_size, self.rl_out_size, hyp["q_embed_dim"], hyp, extraction=extraction)
            print('Using transformer module')

        if 'question_injection_position' in hyp and hyp["question_injection_position"] != 0:
            print('IR model')
        else:     
            print('Non IR model')

    def forward(self, img, qst_idxs, qst_lengths):
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
        
        qst = self.text(qst_idxs, qst_lengths)
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
        
