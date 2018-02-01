"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning"
"""""""""
from __future__ import print_function

import os
import re
import pdb
import json
import pickle
import random
import argparse
import numpy as np

import torch

from torch.utils.data import DataLoader
from torchvision import transforms

import train
import extract
import utils
# import transforms
from model import RN
from clevr_dataset_connector import ClevrDataset, ClevrDatasetImages

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

def main(args):
    args.model_dirs = './model_{}_b{}_lr{}'.format(args.model, args.batch_size, args.lr)
    args.features_dirs = './features'
    if not os.path.exists(args.model_dirs):
        os.makedirs(args.model_dirs)
    
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
    clevr_dataset_feat_extraction = ClevrDatasetImages(args.clevr_dir,'val', test_transforms)

    #Initialize Clevr dataset loaders
    clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=args.batch_size,
                            shuffle=True, num_workers=8, collate_fn=utils.collate_samples)
    clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=args.batch_size / 3,
                            shuffle=False, num_workers=8, collate_fn=utils.collate_samples)
    clevr_feat_extraction_loader = DataLoader(clevr_dataset_feat_extraction, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, drop_last=True)
                            
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

    start_epoch = 1
    if args.resume:
        filename = os.path.join(args.model_dirs, args.resume)
        if os.path.isfile(filename):
            print('==> loading checkpoint {}'.format(filename))
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint)
            print('==> loaded checkpoint {}'.format(filename))
            start_epoch = int(re.match(r'.*epoch_(\d+).pth', args.resume).groups()[0]) + 1

    if args.extract_features != None:
        #start features extraction
        extract.start(clevr_feat_extraction_loader, model, args)
    else:
        #start network train
        train.start(start_epoch, clevr_train_loader, clevr_test_loader, model, args)
        
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')
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
    parser.add_argument('--model', type=str, choices=['original','ir'], default='original',
                    	help='which model is used to train the network')
    parser.add_argument('--extract-features', type=str, default=None,
                    	help='layer of the RN from which features are extracted')
    parser.add_argument('--invert-questions', action='store_true', default=False,
                        help='invert the question word indexes for LSTM processing')

    args = parser.parse_args()
    main(args)
