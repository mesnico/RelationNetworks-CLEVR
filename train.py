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
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, trange
from early_stop import EarlyStopping

import utils
import math
from clevr_dataset_connector import ClevrDataset, ClevrDatasetStateDescription
from model import RN

import ray
from ray import tune
from ray.tune import Trainable, TrainingResult, register_trainable, run_experiments, Experiment
from ray.tune.async_hyperband import AsyncHyperBandScheduler
from ray.tune.hyperband import HyperBandScheduler
from ray.tune.util import pin_in_object_store, get_pinned_object
import config

import pdb
DEBUG = True

def train(data, model, optimizer, epoch, args):
    model.train()

    avg_loss = 0.0
    global_avg_loss = 0.0
    n_batches = 0
    #progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(data):
        img, qst, label = utils.load_tensor_data(sample_batched, args.cuda, args.invert_questions)

        # forward and backward pass
        optimizer.zero_grad()
        output = model(img, qst)
        loss = F.nll_loss(output, label)
        loss.backward()

        # Gradient Clipping
        if args.clip_norm:
            clip_grad_norm(model.parameters(), args.clip_norm)

        optimizer.step()

        # Show progress
        #progress_bar.set_postfix(dict(loss=loss.data[0]))
        avg_loss += loss.data[0]
        global_avg_loss += loss.data[0]
        n_batches += 1

        if batch_idx % args.log_interval == 0:
            avg_loss /= n_batches
            processed = batch_idx * args.batch_size
            n_samples = len(data) * args.batch_size
            progress = float(processed) / n_samples
            #print('Train Epoch: {} [{}/{} ({:.0%})] Train loss: {}'.format(
            #    epoch, processed, n_samples, progress, avg_loss))
            avg_loss = 0.0
            n_batches = 0

    global_avg_loss /= len(data)
    return global_avg_loss

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
    #progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(data):
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
            #progress_bar.set_postfix(dict(acc='{:.2%}'.format(accuracy), inv='{:.2%}'.format(invalids_perc)))
    
    avg_loss /= len(data)
    invalids_perc = invalids / n_samples      
    glob_accuracy = corrects / n_samples

    #print('Test Epoch {}: Accuracy = {:.2%} ({:g}/{}); Invalids = {:.2%} ({:g}/{}); Test loss = {}'.format(epoch, glob_accuracy, corrects, n_samples, invalids_perc, invalids, n_samples, avg_loss))
    for v in class_n_samples.keys():
        accuracy = 0
        invalid = 0
        if class_n_samples[v] != 0:
            accuracy = class_corrects[v] / class_n_samples[v]
            invalid = class_invalids[v] / class_n_samples[v]
        print('{} -- acc: {:.2%} ({}/{}); invalid: {:.2%} ({}/{})'.format(v,accuracy,class_corrects[v],class_n_samples[v],invalid,class_invalids[v],class_n_samples[v]))

    # dump results on file
    #filename = os.path.join(args.test_results_dir, 'test.pickle')
    dump_object = {
        'class_corrects':class_corrects,
        'class_invalids':class_invalids,
        'class_total_samples':class_n_samples,
        'confusion_matrix_target':confusion_matrix_target,
        'confusion_matrix_pred':confusion_matrix_pred,
        'confusion_matrix_labels':sorted_labels,
        'global_accuracy':glob_accuracy
    }
    #pickle.dump(dump_object, open(filename,'wb'))
    return avg_loss, glob_accuracy, dump_object

def reload_loaders(clevr_dataset_train, clevr_dataset_test, train_bs, test_bs, state_description = False):
    if not state_description:
        # Use a weighted sampler for training:
        #weights = clevr_dataset_train.answer_weights()
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        # Initialize Clevr dataset loaders
        clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=train_bs,
                                        shuffle=True, num_workers=8, collate_fn=utils.collate_samples_from_pixels)
        clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs,
                                       shuffle=False, num_workers=8, collate_fn=utils.collate_samples_from_pixels)
    else:
        # Initialize Clevr dataset loaders
        clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=train_bs,
                                        shuffle=True, collate_fn=utils.collate_samples_state_description)
        clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs,
                                       shuffle=False, collate_fn=utils.collate_samples_state_description)
    return clevr_train_loader, clevr_test_loader

def initialize_dataset(clevr_dir, dictionaries, state_description=True):
    if not state_description:
        train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.Pad(8),
                                           transforms.RandomCrop((128, 128)),
                                           transforms.RandomRotation(2.8),  # .05 rad
                                           transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])
                                          
        clevr_dataset_train = ClevrDataset(clevr_dir, True, dictionaries, train_transforms)
        clevr_dataset_test = ClevrDataset(clevr_dir, False, dictionaries, test_transforms)
        
    else:
        clevr_dataset_train = ClevrDatasetStateDescription(clevr_dir, True, dictionaries)
        clevr_dataset_test = ClevrDatasetStateDescription(clevr_dir, False, dictionaries)
    
    return clevr_dataset_train, clevr_dataset_test


#mantains global ids about objects pinned in the store
pinned_obj_dict = {}

class RNTrain(Trainable):
    def _setup(self):
        print('Starting setup with configs: {}'.format(self.config))
        self.start_epoch = 1
        self.epoch = self.start_epoch

        self.args = get_pinned_object(pinned_obj_dict['args'])
        self.dictionaries = get_pinned_object(pinned_obj_dict['dictionaries'])
        self.clevr_dataset_train = get_pinned_object(pinned_obj_dict['clevr_dataset_train'])
        self.clevr_dataset_test = get_pinned_object(pinned_obj_dict['clevr_dataset_test'])

        # Build the model
        self.args.qdict_size = len(self.dictionaries[0])
        self.args.adict_size = len(self.dictionaries[1])

        self.model = RN(self.args, self.config)

        if torch.cuda.device_count() > 1 and self.args.cuda:
            self.model = torch.nn.DataParallel(model)
            self.model.module.cuda()  # call cuda() overridden method

        elif torch.cuda.device_count() == 1 and self.args.cuda:
            self.model.cuda()

        else:
            print('Not using CUDA')

        self.es = EarlyStopping()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, weight_decay=1e-4)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=1e-6, verbose=True)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, self.args.lr_step, gamma=args.lr_gamma)
        self.scheduler.last_epoch = self.start_epoch
        print('Training ({} epochs) is starting...'.format(self.args.epochs))

        self.done = False

    def _train(self):
        bs = self.args.batch_size
        if(((self.args.bs_max > 0 and bs < self.args.bs_max) or self.args.bs_max < 0 ) and (self.epoch % self.args.bs_step == 0 or self.epoch == self.start_epoch)):
            bs = math.floor(self.args.batch_size * (self.args.bs_gamma ** (self.epoch // self.args.bs_step)))
            if bs > self.args.bs_max and self.args.bs_max > 0:
                bs = self.args.bs_max
            self.clevr_train_loader, self.clevr_test_loader = reload_loaders(self.clevr_dataset_train, self.clevr_dataset_test, bs, self.args.test_batch_size, self.config['state_description'])

            #restart optimizer in order to restart learning rate scheduler
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = args.lr
            #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, step, min_lr)
            #print('Dataset reinitialized with batch size {}'.format(bs))
        
        if((self.args.lr_max > 0 and self.scheduler.get_lr()[0]<self.args.lr_max) or self.args.lr_max < 0):
            self.scheduler.step()
                
        print('Current learning rate: {}'.format(self.optimizer.param_groups[0]['lr']))
            
        # TRAIN
        #progress_bar.set_description('TRAIN')
        avg_train_loss = train(self.clevr_train_loader, self.model, self.optimizer, self.epoch, self.args)

        # TEST
        #progress_bar.set_description('TEST')
        avg_test_loss, accuracy, _ = test(self.clevr_test_loader, self.model, self.epoch, self.dictionaries, self.args)

        #check for early-stop
        if self.es.step(avg_test_loss):
            #print('Early-stopping at epoch {}'.format(epoch))
            self.done = True

        now = self.epoch
        self.epoch += 1

        return TrainingResult(
            timesteps_this_iter=1, timesteps_total=now, done=self.done, mean_validation_accuracy=accuracy, mean_loss=avg_train_loss)

    def _save(self, checkpoint_dir):
        filename = 'RN_epoch_{:02d}.pth'.format(self.epoch-1)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save([self.model.state_dict(), self.optimizer.state_dict()], checkpoint_path)
        return checkpoint_path

    def _restore(self, filename):
        #print('==> loading checkpoint {}'.format(filename))
        checkpoint, optimizer_chkp = torch.load(filename)

        #removes 'module' from dict entries, pytorch bug #3805
        checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}

        self.model.load_state_dict(checkpoint)
        self.optimizer.load_state_dict(optimizer_chkp)
        #print('==> loaded checkpoint {}'.format(filename))
        self.start_epoch = int(re.match(r'.*epoch_(\d+).pth', filename).groups()[0]) + 1
        self.epoch = self.start_epoch
        self.scheduler.last_epoch = self.start_epoch

def launch_tensorboard(logdir):
    os.system('tensorboard --logdir=~/ray_results/' + logdir)
    return      

def main(args):
    #load hyperparameters from configuration file
    #with open(args.config) as config_file: 
    #    hyp = json.load(config_file)['hyperparams'][args.model]

    #if args.question_injection >= 0:
    #    hyp['question_injection_position'] = args.question_injection

    #print('Loaded hyperparameters from configuration {}, model: {}: {}'.format(args.config, args.model, hyp))

    #args.model_dirs = './model_{}_drop{}_bstart{}_bstep{}_bgamma{}_bmax{}_lrstart{}_'+ \
    #                  'lrstep{}_lrgamma{}_lrmax{}_invquests-{}_clipnorm{}_glayers{}_qinj{}'
    #args.model_dirs = args.model_dirs.format(
    #                    args.model, hyp['dropouts'], args.batch_size, args.bs_step, args.bs_gamma, 
    #                    args.bs_max, args.lr, args.lr_step, args.lr_gamma, args.lr_max,
    #                    args.invert_questions, args.clip_norm, hyp['g_layers'], hyp['question_injection_position'])
    #if not os.path.exists(args.model_dirs):
    #    os.makedirs(args.model_dirs)
    #create a file in this folder containing the overall configuration
    #args_str = str(args)
    #hyp_str = str(hyp)
    #all_configuration = args_str+'\n\n'+hyp_str
    #filename = os.path.join(args.model_dirs,'config.txt')
    #with open(filename,'w') as config_file:
    #    config_file.write(all_configuration)

    #args.features_dirs = './features'
    args.test_results_dir = './test_results'
    if not os.path.exists(args.test_results_dir):
        os.makedirs(args.test_results_dir)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('Building word dictionaries from all the words in the dataset...')
    dictionaries = utils.build_dictionaries(args.clevr_dir)
    print('Word dictionary completed!')

    print('Initializing CLEVR dataset...')
    clevr_dataset_train, clevr_dataset_test = initialize_dataset(args.clevr_dir, dictionaries, config.hyperparams[args.model]['tune']['state_description'])
    print('CLEVR dataset initialized!')

    ray.init(num_gpus=args.num_gpus)

    print('Pinning global objects into the store...')
    #pin args, hyp, dictionaries and datasets into the store
    pinned_obj_dict['args'] = pin_in_object_store(args)
    pinned_obj_dict['clevr_dataset_train'] = pin_in_object_store(clevr_dataset_train)
    pinned_obj_dict['clevr_dataset_test'] = pin_in_object_store(clevr_dataset_test)
    pinned_obj_dict['dictionaries'] = pin_in_object_store(dictionaries)
    
    '''if args.conv_transfer_learn:
        if os.path.isfile(args.conv_transfer_learn):
            # TODO: there may be problems caused by pytorch issue #3805 if using DataParallel

            print('==> loading conv layer from {}'.format(args.conv_transfer_learn))
            # pretrained dict is the dictionary containing the already trained conv layer
            pretrained_dict = torch.load(args.conv_transfer_learn)

            if torch.cuda.device_count() == 1:
                conv_dict = model.conv.state_dict()
            else:
                conv_dict = model.module.conv.state_dict()
            
            # filter only the conv layer from the loaded dictionary
            conv_pretrained_dict = {k.replace('conv.','',1): v for k, v in pretrained_dict.items() if 'conv.' in k}

            # overwrite entries in the existing state dict
            conv_dict.update(conv_pretrained_dict)

            # load the new state dict
            if torch.cuda.device_count() == 1:
                model.conv.load_state_dict(conv_dict)
                params = model.conv.parameters()
            else:
                model.module.conv.load_state_dict(conv_dict)
                params = model.module.conv.parameters()

            # freeze the weights for the convolutional layer by disabling gradient evaluation
            # for param in params:
            #     param.requires_grad = False

            print("==> conv layer loaded!")
        else:
            print('Cannot load file {}'.format(args.conv_transfer_learn))'''

    #progress_bar = trange(start_epoch, args.epochs + 1)
    if args.test:
        print('TODO')
        # perform a single test
        #print('Testing epoch {}'.format(start_epoch))
        #_, clevr_test_loader = reload_loaders(clevr_dataset_train, clevr_dataset_test, args.batch_size, args.test_batch_size, hyp['state_description'])
        #test(clevr_test_loader, model, start_epoch, dictionaries, args)
    else:
        #TODO: call method from RNTrain class if not using Ray
        #for epoch in progress_bar:
            
        #Using ray
        register_trainable("rn_train", RNTrain)

        # Hyperband early stopping
        hyperband = HyperBandScheduler(
            time_attr="timesteps_total",
            reward_attr="mean_validation_accuracy",
            max_t=args.epochs)

        exp_config = {
            'run': 'rn_train',
            'checkpoint_freq': 5,
            'stop': {'training_iteration':args.epochs},
            'trial_resources': config.hyperparams[args.model]['resources'],
            'repeat': config.hyperparams[args.model]['repeat'],
            'config': config.hyperparams[args.model]['tune']
        }

        if DEBUG:
            exp_config['stop']['training_iteration'] = 2
            exp_config['repeat'] = 2
            exp_config['checkpoint_freq'] = 1

        config_name = "rn_hyperband_{}".format(time.strftime("%Y-%m-%d_%H.%M.%S"))

        #launch tensorboard
        import threading
        t = threading.Thread(target=launch_tensorboard, args=[config_name])
        t.start()

        run_experiments({config_name:exp_config}, scheduler=hyperband)




if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR')
    parser.add_argument('--batch-size', type=int, default=640, metavar='N',
                        help='input batch size for training (default: 640)')
    parser.add_argument('--test-batch-size', type=int, default=640,
                        help='input batch size for training (default: 640)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.000005, metavar='LR',
                        help='learning rate (default: 0.000005)')
    parser.add_argument('--clip-norm', type=int, default=50,
                        help='max norm for gradients; set to 0 to disable gradient clipping (default: 10)')
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
    parser.add_argument('--test', action='store_true', default=False,
                        help='perform only a single test. To use with --resume')
    parser.add_argument('--conv-transfer-learn', type=str,
                    help='use convolutional layer from another training')
    parser.add_argument('--lr-max', type=float, default=0.0005,
                        help='max learning rate')
    parser.add_argument('--lr-gamma', type=float, default=2, 
                        help='increasing rate for the learning rate. 1 to keep LR constant.')
    parser.add_argument('--lr-step', type=int, default=20,
                        help='number of epochs before lr update')
    parser.add_argument('--bs-max', type=int, default=-1,
                        help='max batch-size')
    parser.add_argument('--bs-gamma', type=float, default=1, 
                        help='increasing rate for the batch size. 1 to keep batch-size constant.')
    parser.add_argument('--bs-step', type=int, default=20, 
                        help='number of epochs before batch-size update')
    parser.add_argument('--num-gpus', type=int, default=1, 
                        help='number of available gpus')
    #parser.add_argument('--config', type=str, default='config.json',
    #                    help='configuration file for hyperparameters loading')
    parser.add_argument('--question-injection', type=int, default=-1, 
                        help='At which stage of g function the question should be inserted (0 to insert at the beginning, as specified in DeepMind model, -1 to use configuration value)')
    args = parser.parse_args()
    args.invert_questions = not args.no_invert_questions
    main(args)
