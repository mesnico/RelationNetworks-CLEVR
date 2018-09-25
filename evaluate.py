"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning"
"""""""""
from __future__ import print_function

import argparse
import json
import os
import pickle
import re
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, trange

from itertools import islice
import utils
import math
from clevr_dataset_connector import ClevrDataset, ClevrDatasetStateDescription
from model import RN

import pdb

def test(data, model, epoch, dictionaries, args):
    model.eval()

    # accuracy for every class
    class_corrects = {}
    # for every class, among all the wrong answers, how much are non pertinent
    class_invalids = {}
    # total number of samples for every class
    class_n_samples = {}
    # initialization
    for c in dictionaries[2].values():
        class_corrects[c] = 0
        class_invalids[c] = 0
        class_n_samples[c] = 0

    corrects = 0.0
    invalids = 0.0
    n_samples = 0

    inverted_answ_dict = {v: k for k,v in dictionaries[1].items()}
    sorted_classes = sorted(dictionaries[2].items(), key=lambda x: hash(x[1]) if x[1]!='number' else int(inverted_answ_dict[x[0]]))
    sorted_classes = [c[0]-1 for c in sorted_classes]

    confusion_matrix_target = []
    confusion_matrix_pred = []

    sorted_labels = sorted(dictionaries[1].items(), key=lambda x: x[1])
    sorted_labels = [c[0] for c in sorted_labels]
    sorted_labels = [sorted_labels[c] for c in sorted_classes]

    avg_loss = 0.0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, qst, label = utils.load_tensor_data(sample_batched, args.cuda, args.invert_questions, volatile=True)
        
        output = model(img, qst)
        pred = output.data.max(1)[1]

        loss = F.nll_loss(output, label)

        # compute per-class accuracy
        pred_class = [dictionaries[2][o+1] for o in pred]
        real_class = [dictionaries[2][o+1] for o in label.data]
        for idx,rc in enumerate(real_class):
            class_corrects[rc] += (pred[idx] == label.data[idx])
            class_n_samples[rc] += 1

        for pc, rc in zip(pred_class,real_class):
            class_invalids[rc] += (pc != rc)

        for p,l in zip(pred, label.data):
            confusion_matrix_target.append(sorted_classes.index(l))
            confusion_matrix_pred.append(sorted_classes.index(p))
        
        # compute global accuracy
        corrects += (pred == label.data).sum()
        assert corrects == sum(class_corrects.values()), 'Number of correct answers assertion error!'
        invalids = sum(class_invalids.values())
        n_samples += len(label)
        assert n_samples == sum(class_n_samples.values()), 'Number of total answers assertion error!'
        
        avg_loss += loss.data[0]

        if batch_idx % args.log_interval == 0:
            accuracy = corrects / n_samples
            invalids_perc = invalids / n_samples
            progress_bar.set_postfix(dict(acc='{:.2%}'.format(accuracy), inv='{:.2%}'.format(invalids_perc)))
    
    avg_loss /= len(data)
    invalids_perc = invalids / n_samples      
    accuracy = corrects / n_samples

    print('Test Epoch {}: Accuracy = {:.2%} ({:g}/{}); Invalids = {:.2%} ({:g}/{}); Test loss = {}'.format(epoch, accuracy, corrects, n_samples, invalids_perc, invalids, n_samples, avg_loss))
    for v in class_n_samples.keys():
        accuracy = 0
        invalid = 0
        if class_n_samples[v] != 0:
            accuracy = class_corrects[v] / class_n_samples[v]
            invalid = class_invalids[v] / class_n_samples[v]
        print('{} -- acc: {:.2%} ({}/{}); invalid: {:.2%} ({}/{})'.format(v,accuracy,class_corrects[v],class_n_samples[v],invalid,class_invalids[v],class_n_samples[v]))

    # dump results on file
    filename = os.path.join(args.test_results_dir, 'test.pickle')
    dump_object = {
        'class_corrects':class_corrects,
        'class_invalids':class_invalids,
        'class_total_samples':class_n_samples,
        'confusion_matrix_target':confusion_matrix_target,
        'confusion_matrix_pred':confusion_matrix_pred,
        'confusion_matrix_labels':sorted_labels,
        'global_accuracy':accuracy
    }
    pickle.dump(dump_object, open(filename,'wb'))
    return avg_loss

def reload_loaders(clevr_dataset_test, test_bs, state_description = False):
    if not state_description:
        # Use a weighted sampler for training:
        #weights = clevr_dataset_train.answer_weights()
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        # Initialize Clevr dataset loaders
        clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs,
                                       shuffle=False, num_workers=8, collate_fn=utils.collate_samples_from_pixels)
    else:
        # Initialize Clevr dataset loaders
        clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs,
                                       shuffle=False, num_workers=4, collate_fn=utils.collate_samples_state_description)
    return clevr_test_loader

def initialize_dataset(clevr_dir, dictionaries, state_description=True):
    if not state_description:
        test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])
                                          
        clevr_dataset_test = ClevrDataset(clevr_dir, False, dictionaries, test_transforms)
        
    else:
        clevr_dataset_test = ClevrDatasetStateDescription(clevr_dir, False, dictionaries)
    
    return clevr_dataset_test 
        

class Evaluator():
    def __init__(self, args):
        #load hyperparameters from configuration file
        with open(args.config) as config_file: 
            hyp = json.load(config_file)['hyperparams'][args.model]
        #override configuration dropout
        if args.dropout > 0:
            hyp['dropout'] = args.dropout
        if args.question_injection >= 0:
            hyp['question_injection_position'] = args.question_injection

        print('Loaded hyperparameters from configuration {}, model: {}: {}'.format(args.config, args.model, hyp))

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.invert_questions = not args.no_invert_questions

        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        print('Building word dictionaries from all the words in the dataset...')
        self.dictionaries = utils.build_dictionaries(args.clevr_dir)
        print('Word dictionary completed!')

        print('Initializing CLEVR dataset...')
        clevr_dataset_test  = initialize_dataset(args.clevr_dir, self.dictionaries, hyp['state_description'])
        print('CLEVR dataset initialized!')

        # Build the model
        args.qdict_size = len(self.dictionaries[0])
        args.adict_size = len(self.dictionaries[1])

        self.model = RN(args, hyp)

        if torch.cuda.device_count() > 1 and args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            self.model.module.cuda()  # call cuda() overridden method

        if args.cuda:
            self.model.cuda()

        if args.resume:
            filename = args.resume
            if os.path.isfile(filename):
                print('==> loading checkpoint {}'.format(filename))
                if args.cuda:
                    checkpoint = torch.load(filename)
                else:
                    #map loaded checkpoint onto the CPU
                    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

                #removes 'module' from dict entries, pytorch bug #3805
                if (torch.cuda.device_count() == 1 or not args.cuda) and any(k.startswith('module.') for k in checkpoint.keys()):
                    checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}
                if torch.cuda.device_count() > 1 and not any(k.startswith('module.') for k in checkpoint.keys()):
                    checkpoint = {'module.'+k: v for k,v in checkpoint.items()}

                self.model.load_state_dict(checkpoint)
                print('==> loaded checkpoint {}'.format(filename))

        self.clevr_test_loader = reload_loaders(clevr_dataset_test, 1, hyp['state_description'])    
        self.args = args
        self.answ_reverse_dict = {v:k for k,v in self.dictionaries[1].items()}

    def evaluate(self, sample_idx, question=None):
        handwritten_qst = (question!=None)
                    
        self.model.eval()
        sample = next(islice(self.clevr_test_loader, sample_idx, sample_idx+1))
        
        if handwritten_qst:
            sample['question'] = utils.to_dictionary_indexes(self.dictionaries[0], question, True).unsqueeze(0)

        img, qst, label = utils.load_tensor_data(sample, self.args.cuda, self.args.invert_questions, volatile=True)
        
        output = self.model(img, qst)
        pred = output.data.max(1)[1]

        answ = self.answ_reverse_dict[pred[0]+1]
        if handwritten_qst:
            return answ, None
        else:
            return answ, self.answ_reverse_dict[label.data.cpu().numpy()[0]+1]


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str,
                        help='resume from model stored')
    parser.add_argument('--clevr-dir', type=str, default='.',
                        help='base directory of CLEVR dataset')
    parser.add_argument('--model', type=str, default='original-fp',
                        help='which model is used to train the network')
    parser.add_argument('--no-invert-questions', action='store_true', default=False,
                        help='invert the question word indexes for LSTM processing')
    parser.add_argument('--dropout', type=float, default=-1,
                        help='dropout rate. -1 to use value from configuration')
    parser.add_argument('--config', type=str, default='config.json',
                        help='configuration file for hyperparameters loading')
    parser.add_argument('--question-injection', type=int, default=-1, 
                        help='At which stage of g function the question should be inserted (0 to insert at the beginning, as specified in DeepMind model, -1 to use configuration value)')
    parser.add_argument('--write-question', action='store_true', default=False,
                        help='Hand write the question')

    args = parser.parse_args()
    
    #sample evaluation
    qid = input('Enter question ID from test set: ')
    ev = Evaluator(args)
    while(True):
        
        if args.write_question:
            qst = input('Your question: ')
        else:
            qst = None
        try:
            answ = ev.evaluate(int(qid), qst)
            print(answ)
        except ValueError:
            print('Please enter a numeric index for question ID')
        except KeyError as e:
            print('Word not existing in dictionary: {}'.format(str(e)))
            
        
