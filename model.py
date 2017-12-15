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
  

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name = name

    def save_model(self, epoch, model_dir='model'):
        fname = 'epoch_{}_{:02d}.pth'.format(self.name, epoch)
        torch.save(self.state_dict(), os.path.join(model_dir, fname))


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')

        self.args = args
        self.qembed_size = 32
        self.hidden_size = 128
        
        self.wembedding = nn.Embedding(args.qdict_size+1, self.qembed_size, padding_idx=0)  #word embeddings have size 32. Indexes 0 output 0 vectors (variable len questions)
        self.lstm = nn.LSTM(self.qembed_size, self.hidden_size)  # Input dim is 32, output dim is the question embedding
        # initialize the hidden and cell states of the lstm.
        
        self.conv = ConvInputModel()
        self.conv_out_size = 8	#calculated from actual convolutional layer parameters
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+self.hidden_size, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)
        self.f_fc2 = nn.Linear(256, 256)
        self.f_fc3 = nn.Linear(256, args.adict_size)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/self.conv_out_size - self.conv_out_size/2)/(self.conv_out_size/2.), 
                    (i%self.conv_out_size - self.conv_out_size/2)/(self.conv_out_size/2.)]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, self.conv_out_size**2, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, self.conv_out_size**2, 2))
        for i in range(self.conv_out_size**2):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))

    def init_lstm_hidden(self):
        hidden_state = torch.randn(1, self.args.batch_size, self.hidden_size)
        cell_state = torch.randn(1, self.args.batch_size, self.hidden_size)
        if self.args.cuda:
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()
        hidden_state = Variable(hidden_state)
        cell_state = Variable(cell_state)

        return (hidden_state, cell_state)

    def forward(self, img, qst_idxs):
        #reset the lstm hidden state
        self.hidden = self.init_lstm_hidden()

        #calculate question embeddings
        #pdb.set_trace()
        wembed = self.wembedding(qst_idxs)

        wembed = wembed.permute(1,0,2)   #in lstm minibatches are in the 2-nd dimension
        #wembed = wembed.view(len(qst_idxs[0]), self.args.batch_size , -1)
        #pdb.set_trace()

        _, self.hidden = self.lstm(wembed, self.hidden)
        qst = self.hidden[1] # last layer of the lstm. qst = (64 x 128)
        #take the hidden state at the last LSTM layer (as of now, there is only one LSTM layer)
        qst = qst[-1]

        x = self.conv(img) ## x = (64 x 24 x 8 x 8)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 64 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        # x_flat = (64 x 64 x 24+2)
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)   #(64 x 1 x 128)
        qst = qst.repeat(1,self.conv_out_size**2,1) #(64 x 64 x 128)
        qst = torch.unsqueeze(qst, 2)   #(64 x 64 x 1 x 128)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64 x 1 x 64 x 26)
        x_i = x_i.repeat(1,self.conv_out_size**2,1,1) # (64 x 64 x 64 x 26)
        x_j = torch.unsqueeze(x_flat,2) # (64 x 64 x 1 x 26)
        x_j = torch.cat([x_j, qst],3)
        x_j = x_j.repeat(1,1,self.conv_out_size**2,1) # (64 x 64 x 64 x 26+128)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64 x 64 x 64 x 2*26+128)
        
        # reshape for passing through network
        x_ = x_full.view(mb*d*d*d*d,2*26+self.hidden_size)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb,d*d*d*d,256)
        x_g = x_g.sum(1).squeeze(1)
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.dropout1(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = self.dropout2(x_f)
        x_f = self.f_fc3(x_f)
        #x_f = F.relu(x_f)
        return F.log_softmax(x_f)
