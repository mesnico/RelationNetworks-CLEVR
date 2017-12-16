import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


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
    def __init__(self, in_size, emb_size=32, hidden_size=128, cuda=False):
        super(QuestionEmbedModel, self).__init__()
        self.cuda = cuda
        self.hidden_size = hidden_size
        
        self.wembedding = nn.Embedding(in_size + 1, emb_size, padding_idx=0)  #word embeddings have size 32. Indexes 0 output 0 vectors (variable len questions)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)  # Input dim is 32, output dim is the question embedding
        
    def forward(self, question):
        # initialize the lstm hidden state
        batch_size = question.size()[0]
        self.hidden = (Variable(torch.zeros(1, batch_size, self.hidden_size)),
                       Variable(torch.zeros(1, batch_size, self.hidden_size)))
        if self.cuda: 
            self.hidden = tuple(t.cuda() for t in self.hidden)
        
        #calculate question embeddings
        wembed = self.wembedding(question)

        # wembed = wembed.permute(1,0,2)   #in lstm minibatches are in the 2-nd dimension
        #wembed = wembed.view(len(qst_idxs[0]), self.args.batch_size , -1)

        _, self.hidden = self.lstm(wembed, self.hidden)
        qst = self.hidden[1] # last layer of the lstm. qst = (B x 128)
        #take the hidden state at the last LSTM layer (as of now, there is only one LSTM layer)
        qst = qst[-1]
        
        return qst
        

class RelationalLayerModel(nn.Module):
    def __init__(self, in_size, out_size, conv_out_size, batch_size, cuda=False):
        super(RelationalLayerModel, self).__init__()
        
        self.conv_out_size = conv_out_size
        
        self.g_fc1 = nn.Linear(in_size, 256)
        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_fc3 = nn.Linear(256, out_size)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        # prepare coord tensor
        def build_coord_tensor(s, b):
            a = torch.arange(0, s**2)
            x = (a/s - s/2)/(s/2.)
            y = (a%s - s/2)/(s/2.)
            return torch.stack((x,y), dim=1).repeat(b, 1, 1)
        
        # broadcast to all batches
        # TODO: upgrade pytorch and use broadcasting
        coord_tensor = build_coord_tensor(self.conv_out_size, batch_size)
        self.coord_tensor = Variable(coord_tensor)
        
        if cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        
    def forward(self, x, qst):
        # x = (B x 24 x 8 x 8)
        # qst = (B x 128)
        """g"""
        mb, n_channels, d, _ = x.size()
        qst_size = qst.size()[1]

        x_flat = x.view(mb, n_channels, d*d).permute(0,2,1)# (B x 64 x 24)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor], 2) # (B x 64 x 24+2)
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)                      # (B x 1 x 128)
        qst = qst.repeat(1, self.conv_out_size**2, 1)      # (B x 64 x 128)
        qst = torch.unsqueeze(qst, 2)                      # (B x 64 x 1 x 128)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)                   # (B x 1 x 64 x 26)
        x_i = x_i.repeat(1, self.conv_out_size**2, 1, 1)   # (B x 64 x 64 x 26)
        x_j = torch.unsqueeze(x_flat, 2)                   # (B x 64 x 1 x 26)
        x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, self.conv_out_size**2, 1)   # (B x 64 x 64 x 26+128)
        
        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)                  # (B x 64 x 64 x 2*26+128)
        
        # reshape for passing through network
        x_ = x_full.view(mb * d**4, 2*26 + qst_size)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb, d**4, 256)
        x_g = x_g.sum(1).squeeze(1)
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.dropout1(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = self.dropout2(x_f)
        x_f = self.f_fc3(x_f)

        return F.log_softmax(x_f)


class BasicModel(nn.Module):
    def __init__(self, name):
        super(BasicModel, self).__init__()
        self.name = name

    def save_model(self, epoch, model_dir='model'):
        fname = 'epoch_{}_{:02d}.pth'.format(self.name, epoch)
        torch.save(self.state_dict(), os.path.join(model_dir, fname))
    

class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__('RN')
        
        self.hidden_size = 128
        self.text = QuestionEmbedModel(args.qdict_size,
            hidden_size=self.hidden_size, cuda=args.cuda)
        
        # calculated from actual convolutional layer parameters
        self.conv_out_size = 8
        # (number of filters per object + coordinate of object)*2 + question vector
        self.relat_input_size = (24 + 2)*2 + self.hidden_size
        self.relat_output_size = args.adict_size
        
        self.conv = ConvInputModel()
        self.relat = RelationalLayerModel(self.relat_input_size,
            self.relat_output_size, self.conv_out_size, args.batch_size, args.cuda)
        
        #if self.parallel:
        #    self.conv = nn.DataParallel(self.conv, dim=0)
        #    self.text = nn.DataParallel(self.text, dim=1)
        #    self.relat = nn.DataParallel(self.relat, dim=0)

    def forward(self, img, qst_idxs):
        x = self.conv(img)
        qst = self.text(qst_idxs)
        y = self.relat(x, qst)
        return y
        
