"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
"""""""""
from __future__ import print_function
import argparse
import os
#import cPickle as pickle
import pickle
import random
import numpy as np
import json
import pdb
from torch.utils.data import DataLoader
from clevr_dataset_connector import ClevrDataset
from torchvision.transforms import Compose
import transforms
import utils

#import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from model import RN

'''
    Build questions and answers dictionaries over the entire dataset

'''
def build_dictionaries():
    quest_to_ix = {}
    answ_to_ix = {}
    json_train_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
    #load all words from all training data
    with open(json_train_filename,"r") as f:
        questions = json.load(f)['questions']
        for q in questions:
            question = q['question'].split()
            answer = q['answer']
            #pdb.set_trace()
            for word in question:
                w = word.lower()	#no distinction between initial words and the others
                if w not in quest_to_ix:
                    quest_to_ix[w] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding
            
            a = answer.lower()
            if a not in answ_to_ix:
                    answ_to_ix[a] = len(answ_to_ix)+1
            
    return (quest_to_ix, answ_to_ix)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.00025, metavar='LR',
                    help='learning rate (default: 0.00025)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--clevr-dir', type=str, default='.',
                    help='base directory of CLEVR dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model_dirs = './model'
clevr_dir = args.clevr_dir

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print('Building word dictionaries from all the words in the dataset...')
dictionaries = build_dictionaries()

print('Word dictionary completed!')

print('Initializing CLEVR dataset...')
composed_transforms = Compose([ transforms.Resize((128,128)),
                                transforms.Pad(8),
                                transforms.RandomCrop((128,128)),
                                transforms.RandomRotate(2.8),
                                transforms.ToTensor()])
'''composed_transforms = Compose([ transforms.Resize((128,128)),
                                transforms.ToTensor()])'''
clevr_dataset_train = ClevrDataset(clevr_dir, True, dictionaries, composed_transforms)
clevr_dataset_test = ClevrDataset(clevr_dir, False, dictionaries, composed_transforms) 

print('CLEVR dataset initialized!')   

#Build the model
args.qdict_size = len(dictionaries[0])
args.adict_size = len(dictionaries[1])
model = RN(args)

bs = args.batch_size
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_qst = torch.LongTensor(bs, 11)
label = torch.LongTensor(bs)

if args.cuda:
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

#Initialize Clevr dataset loaders
clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=bs,
                        shuffle=True, num_workers=8, collate_fn=utils.collate_samples, drop_last=True)
clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=bs,
                        shuffle=False, num_workers=8, collate_fn=utils.collate_samples, drop_last=True)

def load_tensor_data(data_batch):
    img = data_batch['image']
    qst = data_batch['question']
    ans = data_batch['answer']

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)

    
def train(epoch):
    model.train()

    for batch_idx, sample_batched in enumerate(clevr_train_loader):   
        #sample_batched['answer'] = torch.IntTensor(args.batch_size, 1).random_(1,len(dictionaries[1]))
        #sample_batched['question'] = torch.IntTensor(args.batch_size, 18).random_(1,len(dictionaries[0]))     
        load_tensor_data(sample_batched)
        accuracy = model.train_(input_img, input_qst, label-1)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Train accuracy: {:.0f}%\n'.format(epoch, batch_idx * bs, len(clevr_dataset_train), \
                 100. * batch_idx * bs/ len(clevr_dataset_train), accuracy))
            

def test(epoch):
    model.eval()

    accuracy = []
    for batch_idx, sample_batched in enumerate(clevr_test_loader):
        load_tensor_data(sample_batched)

        accuracy.append(model.test_(input_img, input_qst, label-1))
    mean = sum(accuracy) / len(accuracy)
    
    print('\n Test set: accuracy: {:.0f}%; Best batch accuracy: {:.0f}%\n'.format(mean,max(accuracy)))

'''def train(epoch):
    model.train()

    for batch_idx in range(60):        
        sample_batched = {}
        sample_batched['image'] = torch.Tensor(32,3,128,128).uniform_(0,1)
        sample_batched['question'] = torch.IntTensor(32,17).random_(1,200)
        sample_batched['answer'] = torch.IntTensor(32,1).random_(1,30)
        load_tensor_data(sample_batched)
        accuracy = model.train_(input_img, input_qst, label-1)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Train accuracy: {:.0f}%\n'.format(epoch, batch_idx * bs, 60, \
                 100. * batch_idx * bs/ 60, accuracy))
            
s_batched = {}
s_batched['image'] = torch.Tensor(32,3,128,128).uniform_(0,1)
s_batched['question'] = torch.IntTensor(32,17).random_(1,200)
s_batched['answer'] = torch.IntTensor(32,1).random_(1,30)

def test(epoch):
    model.eval()

    accuracy = []
    for batch_idx in range(1):        
        
        load_tensor_data(s_batched)
        
        accuracy.append(model.test_(input_img, input_qst, label-1))
        pdb.set_trace()
    mean = sum(accuracy) / len(accuracy)
    print('\n Test set: accuracy: {:.0f}%; Best batch accuracy: {:.0f}%\n'.format(mean,max(accuracy)))'''

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

print('Training ({} epochs) is starting...'.format(args.epochs))
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    model.save_model(epoch)
