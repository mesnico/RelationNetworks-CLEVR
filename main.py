"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning"
"""""""""
from __future__ import print_function

import os
import pdb
import json
import pickle
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms 

from tqdm import tqdm, trange

import utils
# import transforms
from model import RN
from clevr_dataset_connector import ClevrDataset

'''
    Build questions and answers dictionaries over the entire dataset
'''
def build_dictionaries(clevr_dir):
    cached_dictionaries = os.path.join(clevr_dir, 'questions', 'CLEVR_built_dictionaries.pkl')
    if os.path.exists(cached_dictionaries):
        with open(cached_dictionaries, 'rb') as f:
            return pickle.load(f)
            
    quest_to_ix = {}
    answ_to_ix = {}
    json_train_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
    #load all words from all training data
    with open(json_train_filename, "r") as f:
        questions = json.load(f)['questions']
        for q in tqdm(questions):
            question = utils.tokenize(q['question'])
            answer = q['answer']
            #pdb.set_trace()
            for word in question:
                if word not in quest_to_ix:
                    quest_to_ix[word] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding
            
            a = answer.lower()
            if a not in answ_to_ix:
                    answ_to_ix[a] = len(answ_to_ix)+1

    ret = (quest_to_ix, answ_to_ix)    
    with open(cached_dictionaries, 'wb') as f:
        pickle.dump(ret, f)

    return ret


def load_tensor_data(data_batch, cuda, volatile=False):
    # prepare input
    var_kwargs = dict(volatile=True) if volatile else dict(requires_grad=False)
    
    img = Variable(data_batch['image'], **var_kwargs)
    qst = Variable(data_batch['question'], **var_kwargs)
    label = Variable(data_batch['answer'], **var_kwargs)
    if cuda:
       img, qst, label = img.cuda(), qst.cuda(), label.cuda()
       
    label = (label - 1).squeeze(1)
    return img, qst, label


def train(data, model, optimizer, epoch, args):
    model.train()
    
    avg_loss = 0.0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, qst, label = load_tensor_data(sample_batched, args.cuda)
        
        # forward and backward pass
        optimizer.zero_grad()
        output = model(img, qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.data[0]
        
        if batch_idx % args.log_interval == 0:
            avg_loss /= args.log_interval
            progress_bar.set_postfix(dict(loss=avg_loss))
            processed = batch_idx * args.batch_size
            n_samples = len(data) * args.batch_size
            progress = float(processed) / n_samples
            print('Train Epoch: {} [{}/{} ({:.0%})] Train loss: {}'.format(
                epoch, processed, n_samples, progress, avg_loss))
            avg_loss = 0.0
            

def test(data, model, epoch, args):
    model.eval()

    corrects = 0.0
    n_samples = 0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, qst, label = load_tensor_data(sample_batched, args.cuda, volatile=True)
        
        output = model(img, qst)
        
        # compute accuracy
        pred = output.data.max(1)[1]
        corrects += (pred == label.data).sum()
        n_samples += len(label)
        
        if batch_idx % args.log_interval == 0:
            accuracy = corrects / n_samples
            progress_bar.set_postfix(dict(acc='{:.2%}'.format(accuracy)))
            
    accuracy = corrects / n_samples
    print('Test Epoch {}: Accuracy = {:.2%} ({:g}/{})'.format(epoch, accuracy, corrects, n_samples))


def main(args):
    model_dirs = './model_b{}_lr{}'.format(args.batch_size, args.lr)
    if not os.path.exists(model_dirs):
        os.makedirs(model_dirs)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('Building word dictionaries from all the words in the dataset...')
    dictionaries = build_dictionaries(args.clevr_dir)
    print('Word dictionary completed!')

    print('Initializing CLEVR dataset...')
    train_transforms = transforms.Compose([ transforms.Resize((128, 128)),
                                            transforms.Pad(8),
                                            transforms.RandomCrop((128, 128)),
                                            transforms.RandomRotation(2.8), # .05 rad
                                            transforms.ToTensor()])
    test_transforms = transforms.Compose([ transforms.Resize((128, 128)),
                                            transforms.ToTensor()])

    clevr_dataset_train = ClevrDataset(args.clevr_dir, True, dictionaries, train_transforms)
    clevr_dataset_test = ClevrDataset(args.clevr_dir, False, dictionaries, test_transforms) 

    #Initialize Clevr dataset loaders
    clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=args.batch_size,
                            shuffle=True, num_workers=8, collate_fn=utils.collate_samples)
    clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, collate_fn=utils.collate_samples)
                            
    print('CLEVR dataset initialized!')   

    #Build the model
    args.qdict_size = len(dictionaries[0])
    args.adict_size = len(dictionaries[1])
    model = RN(args)
    
    
    if torch.cuda.device_count() > 1 and args.cuda:
        model = torch.nn.DataParallel(model)
        model.module.cuda() # call cuda() overridden method 
        
    if args.cuda:
        model.cuda()

    if args.resume:
        filename = os.path.join(model_dirs, args.resume)
        if os.path.isfile(filename):
            print('==> loading checkpoint {}'.format(filename))
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint)
            print('==> loaded checkpoint {}'.format(filename))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Training ({} epochs) is starting...'.format(args.epochs))
    progress_bar = trange(1, args.epochs + 1) 
    for epoch in progress_bar:
        # TRAIN
        progress_bar.set_description('TRAIN')
        train(clevr_train_loader, model, optimizer, epoch, args)
        # TEST
        progress_bar.set_description('TEST')
        test(clevr_test_loader, model, epoch, args)
        # SAVE MODEL
        fname = 'RN_epoch_{:02d}.pth'.format(epoch)
        torch.save(model.state_dict(), os.path.join(model_dirs, fname))
        
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
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
    main(args)

