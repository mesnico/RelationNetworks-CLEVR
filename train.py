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
import glob

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
import custom_lr_schedulers
from clevr_dataset_connector import ClevrDataset, ClevrDatasetStateDescription
from model import RN

import pdb

ALL_IN_MEMORY_CACHE = True
#torch.backends.cudnn.enabled = False

def train(data, model, optimizer, epoch, args):
    model.train()

    avg_loss = 0.0
    n_batches = 0
    minibatch_loss_sum = 0
    progress_bar = tqdm(data)
    optimizer.zero_grad()
    for minibatch_idx, sample_batched in enumerate(progress_bar):
        img, qst, label, qst_len = utils.load_tensor_data(sample_batched, args.cuda)
        # forward and backward pass

        output = model(img, qst, qst_len)
        loss = F.nll_loss(output, label)
        loss /= args.minibatches
        minibatch_loss_sum += loss
        loss.backward()

        # Gradient Clipping
        if args.clip_norm:
            clip_grad_norm(model.parameters(), args.clip_norm)

        if (minibatch_idx+1) % args.minibatches == 0:
            #all minibatches have been accumulated. Zero the grad
            optimizer.step()
            optimizer.zero_grad()

            # Show progress
            progress_bar.set_postfix(dict(loss=minibatch_loss_sum.data.item()))
            avg_loss += minibatch_loss_sum.data.item()
            n_batches += 1
            minibatch_loss_sum = 0

        if (minibatch_idx+1) % (args.log_interval*args.minibatches) == 0:
            avg_loss /= n_batches
            processed = minibatch_idx
            n_samples = len(data)
            progress = float(processed) / n_samples
            print('Train Epoch: {} [{}/{} ({:.0%})] Train loss: {}'.format(
                epoch, processed, n_samples, progress, avg_loss))
            avg_loss = 0.0
            n_batches = 0


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
        img, qst, label, qst_len = utils.load_tensor_data(sample_batched, args.cuda)
        with torch.no_grad():
            output = model(img, qst, qst_len)
        pred = output.data.max(1)[1]

        loss = F.nll_loss(output, label)

        # compute per-class accuracy
        pred_class = [dictionaries[2][o.item()+1] for o in pred]
        real_class = [dictionaries[2][o.item()+1] for o in label.data]
        for idx,rc in enumerate(real_class):
            class_corrects[rc] += (pred[idx].item() == label.data[idx].item())
            class_n_samples[rc] += 1

        for pc, rc in zip(pred_class,real_class):
            class_invalids[rc] += (pc != rc)

        for p,l in zip(pred, label.data):
            confusion_matrix_target.append(sorted_classes.index(l))
            confusion_matrix_pred.append(sorted_classes.index(p))
        
        # compute global accuracy
        corrects += (pred == label.data).sum().item()
        assert corrects == sum(class_corrects.values()), 'Number of correct answers assertion error!'
        invalids = sum(class_invalids.values())
        n_samples += len(label)
        assert n_samples == sum(class_n_samples.values()), 'Number of total answers assertion error!'
        
        avg_loss += loss.data.item()

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

def initialize_dataset(clevr_dir, dictionaries, state_description=True, invert_questions=True):
    if not state_description:
        train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.Pad(8),
                                           transforms.RandomCrop((128, 128)),
                                           transforms.RandomRotation(2.8),  # .05 rad
                                           transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])
                                          
        clevr_dataset_train = ClevrDataset(clevr_dir, True, dictionaries, invert_questions, train_transforms, ALL_IN_MEMORY_CACHE)
        clevr_dataset_test = ClevrDataset(clevr_dir, False, dictionaries, invert_questions, test_transforms, ALL_IN_MEMORY_CACHE)
        
    else:
        clevr_dataset_train = ClevrDatasetStateDescription(clevr_dir, True, dictionaries, invert_questions, ALL_IN_MEMORY_CACHE)
        clevr_dataset_test = ClevrDatasetStateDescription(clevr_dir, False, dictionaries, invert_questions, ALL_IN_MEMORY_CACHE)
    
    return clevr_dataset_train, clevr_dataset_test 
        
    


def main(args):
    #load hyperparameters from configuration file
    with open(args.config) as config_file: 
        hyp = json.load(config_file)['hyperparams'][args.model]

    if args.question_injection >= 0:
        hyp['question_injection_position'] = args.question_injection

    print('Loaded hyperparameters from configuration {}, model: {}: {}'.format(args.config, args.model, hyp))

    lr_params = ''
    if 'lr_scheduler' in hyp:
        for k,v in hyp['lr_scheduler'].items():
            lr_params += k+str(v)+'_'
    args.model_dirs = './model_{}{}_drop{}_bs{}_lrstart{}_'+ \
                      '{}invquests-{}_clipnorm{}_glayers{}_qinj{}'
    args.model_dirs = args.model_dirs.format(
                        args.model, '-transf_learn' if args.transfer_learn else '', hyp['dropouts'], args.batch_size, hyp['lr'], lr_params,
                        args.invert_questions, args.clip_norm, hyp['g_layers'], hyp['question_injection_position'])
    if not os.path.exists(args.model_dirs):
        os.makedirs(args.model_dirs)
    #create a file in this folder containing the overall configuration
    args_str = str(args)
    hyp_str = str(hyp)
    all_configuration = args_str+'\n\n'+hyp_str
    filename = os.path.join(args.model_dirs,'config.txt')
    with open(filename,'w') as config_file:
        config_file.write(all_configuration)

    args.features_dirs = './features'
    args.test_results_dir = './test_results'
    if not os.path.exists(args.test_results_dir):
        os.makedirs(args.test_results_dir)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('Cuda visible devices: {}'.format(torch.cuda.device_count()))

    print('Building word dictionaries from all the words in the dataset...')
    dictionaries = utils.build_dictionaries(args.clevr_dir)
    print('Word dictionary completed!')

    print('Initializing CLEVR dataset...')
    clevr_dataset_train, clevr_dataset_test  = initialize_dataset(args.clevr_dir, dictionaries, hyp['state_description'], args.invert_questions)
    print('CLEVR dataset initialized!')

    print('Minibatches have size: {}'.format(args.batch_size))

    # Build the model
    args.qdict_size = len(dictionaries[0])
    args.adict_size = len(dictionaries[1])

    model = RN(args, hyp)

    if torch.cuda.device_count() > 1 and args.cuda:
        model = torch.nn.DataParallel(model)
        model.module.cuda()  # call cuda() overridden method

    if args.cuda:
        model.cuda()

    start_epoch = 1
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyp['lr'], weight_decay=1e-4)

    if 'lr_scheduler' in hyp:
        shyp = hyp['lr_scheduler']
        if shyp['type'] == 'exp_step':
        	scheduler = custom_lr_schedulers.ClampedStepLR(optimizer, shyp['step'], gamma=shyp['gamma'], maximum_lr=shyp['max'])
        elif shyp['type'] == 'cosine_annealing_restarts':
            scheduler = custom_lr_schedulers.CosineAnnealingRestartsLR(optimizer, T=shyp['T'], eta_min=shyp['eta_min'], T_mult=shyp['T_mult'], eta_mult=shyp['eta_mult'])
        elif shyp['type'] == 'exp_step_restarts':
            scheduler = custom_lr_schedulers.ClampedStepRestartsLR(optimizer, shyp['T'], shyp['step'], gamma=shyp['gamma'], maximum_lr=shyp['max'])
        else:
            raise ValueError('LR algorithm not found: {}'.format(sype['type']))

        print('Using {} LR scheduler'.format(shyp['type']))
    else:
        scheduler = None

    # --- resume code ---
    '''def get_latest_pth(path):
        """Returns the name of the latest (most recent) file 
        of the joined path(s)"""
        fullpath = os.path.join(path, *paths)
        files = glob.glob(fullpath)  # You may use iglob in Python3
        if not files:                # I prefer using the negation
            return None                      # because it behaves like a shortcut
        latest_file = max(files, key=os.path.getctime)
        #_, filename = os.path.split(latest_file)
        return filename'''

    if args.resume or args.auto_resume:
        if args.resume:
            filename = args.resume
        elif args.auto_resume:
            #files = glob.glob(args.model_dirs+'/*.pth')
            files = os.listdir(args.model_dirs)
            files = [os.path.join(args.model_dirs,f) for f in files if '.pth' in f]
            filename = max(files, key=os.path.getctime) if files else None
        if filename!=None and os.path.isfile(filename):
            print('==> loading checkpoint {}'.format(filename))
            checkpoint, optimizer_chkp, scheduler_chkp = torch.load(filename)

            #removes 'module' from dict entries, pytorch bug #3805
            checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}

            model.load_state_dict(checkpoint)
            optimizer.load_state_dict(optimizer_chkp)
            scheduler.load_state_dict(scheduler_chkp)

            print('==> loaded checkpoint {}'.format(filename))
            start_epoch = int(re.match(r'.*epoch_(\d+).pth', filename).groups()[0]) + 1        

    # --- convolutional transfer learn code ---
    if args.transfer_learn and not (args.resume or args.auto_resume):
        if os.path.isfile(args.transfer_learn):
            # TODO: there may be problems caused by pytorch issue #3805 if using DataParallel

            print('==> loading conv and RN layers from {}'.format(args.transfer_learn))
            # pretrained dict is the dictionary containing the already trained conv layer
            pretrained_dict = torch.load(args.transfer_learn)

            if torch.cuda.device_count() == 1:
                conv_dict = model.conv.state_dict()
                rl_dict = model.rl.state_dict()
            else:
                conv_dict = model.module.conv.state_dict()
                rl_dict = model.module.rl.state_dict()
            
            # filter only the conv layer from the loaded dictionary
            conv_pretrained_dict = {k.replace('module.','').replace('conv.','',1): v for k, v in pretrained_dict.items() if 'conv.' in k}

            # get the weights from the first 2 layers of g
            rl_pretrained_dict = {k.replace('module.','').replace('rl.','',1): v for k, v in pretrained_dict.items() if 'rl.' in k and 'f_fc' not in k and '.2.' not in k and '.3.' not in k}

            # overwrite entries in the existing state dict
            conv_dict.update(conv_pretrained_dict)
            rl_dict.update(rl_pretrained_dict)

            # load the new state dict
            if torch.cuda.device_count() == 1:
                model.conv.load_state_dict(conv_dict)
                model.rl.load_state_dict(rl_dict)
                #params = model.conv.parameters()
            else:
                model.module.conv.load_state_dict(conv_dict)
                model.module.rl.load_state_dict(rl_dict)
                #params = model.module.conv.parameters()

            # freeze the weights for the convolutional layer by disabling gradient evaluation
            # for param in params:
            #     param.requires_grad = False

            print("==> conv and RL layer loaded!")
        else:
            print('Cannot load file {}'.format(args.transfer_learn))

    progress_bar = trange(start_epoch, args.epochs + 1)
    if args.test:
        # perform a single test
        print('Testing epoch {}'.format(start_epoch))
        _, clevr_test_loader = reload_loaders(clevr_dataset_train, clevr_dataset_test, args.batch_size, args.test_batch_size, hyp['state_description'])
        test(clevr_test_loader, model, start_epoch, dictionaries, args)
    else:
        bs = args.batch_size

        # perform a full training

        mod_patience = args.patience // args.validation_interval
        print('Patience is {} epochs; with validation interval of {} it is set to {}'.format(args.patience, args.validation_interval, mod_patience))
        es = EarlyStopping(patience=mod_patience)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=1e-6, verbose=True)
        
        #scheduler.last_epoch = start_epoch
        print('Training ({} epochs) is starting...'.format(args.epochs))

        clevr_train_loader, clevr_test_loader = reload_loaders(clevr_dataset_train, clevr_dataset_test, args.batch_size, args.test_batch_size, hyp['state_description'])

        for epoch in progress_bar:
            
            if scheduler != None:
                scheduler.step()
                    
            print('Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))
                
            # TRAIN
            progress_bar.set_description('TRAIN')
            train(clevr_train_loader, model, optimizer, epoch, args)

            # TEST
            if epoch % args.validation_interval == 0:
                progress_bar.set_description('TEST')
                avg_loss = test(clevr_test_loader, model, epoch, dictionaries, args)

                #check for early-stop
                if es.step(avg_loss):
                    print('Early-stopping at epoch {}'.format(epoch))
                    break

            # SAVE MODEL
            filename = 'RN_epoch_{:02d}.pth'.format(epoch)
            torch.save([model.state_dict(), optimizer.state_dict(), scheduler.state_dict()], os.path.join(args.model_dirs, filename))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR')
    parser.add_argument('--batch-size', type=int, default=640, metavar='N',
                        help='input batch size for training (default: 640)')
    parser.add_argument('--test-batch-size', type=int, default=640,
                        help='input batch size for training (default: 640)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--clip-norm', type=int, default=50,
                        help='max norm for gradients; set to 0 to disable gradient clipping (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume from model stored')
    parser.add_argument('--clevr-dir', type=str, default='.',
                        help='base directory of CLEVR dataset')
    parser.add_argument('--model', type=str, default='original-fp',
                        help='which model is used to train the network')
    parser.add_argument('--no-invert-questions', action='store_true', default=False,
                        help='invert the question word indexes for LSTM processing')
    parser.add_argument('--test', action='store_true', default=False,
                        help='perform only a single test. To use with --resume')
    parser.add_argument('--transfer-learn', type=str,
                    help='use layers from another training')
    parser.add_argument('--config', type=str, default='config.json',
                        help='configuration file for hyperparameters loading')
    parser.add_argument('--question-injection', type=int, default=-1, 
                        help='At which stage of g function the question should be inserted (0 to insert at the beginning, as specified in DeepMind model, -1 to use configuration value)')
    parser.add_argument('--validation-interval', type=int, default=5, 
                        help='number of epochs before next validation')
    parser.add_argument('--patience', type=int, default=50, 
                        help='number of epochs before stopping training if no improvements occur')
    parser.add_argument('--auto-resume', action='store_true', default=False,
                        help='Auto resume from folder pertaining to the current experiment')
    parser.add_argument('--minibatches', type=int, default=1, 
                        help='number of minibatches in every batch. Needed for batch-accumulation')
    args = parser.parse_args()
    args.invert_questions = not args.no_invert_questions

    #TODO: args.batch_size becomes actually the minibatch size. Global refactor needed
    args.batch_size //= args.minibatches

    assert not (args.auto_resume==True and args.resume!=None), '--auto-resume and --resume options are mutually exclusive' 
    main(args)
