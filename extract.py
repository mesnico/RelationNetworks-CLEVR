"""
Pytorch implementation of "A simple neural network module for relational reasoning"
"""
from __future__ import print_function

import argparse
import os
import pickle
import json

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import utils
from clevr_dataset_connector import ClevrDatasetImages, ClevrDatasetImagesStateDescription
from model import RN

def extract_features_rl(data, quest_inject_index, max_features_file, avg_features_file, model, args):
    lay, io = args.layer.split(':') #TODO getting extraction layer from quest_inject_index, lay is unused

    maxf = []
    avgf = []

    def hook_function(m, i, o):
        nonlocal maxf, avgf
        '''print(
            'm:', type(m),
            '\ni:', type(i),
                '\n   len:', len(i),
                '\n   type:', type(i[0]),
                '\n   data size:', i[0].data.size(),
                '\n   data type:', i[0].data.type(),
            '\no:', type(o),
                '\n   data size:', o.data.size(),
                '\n   data type:', o.data.type(),
        )'''
        if io == 'i':
            z = i[0]
        else:
            z = o
        # aggregate features
        d4_combinations = z.size()[0] // args.batch_size
        x_ = z.view(args.batch_size, d4_combinations, z.size()[1])
        maxf = x_.max(1)[0].squeeze()
        avgf = x_.mean(1).squeeze()

        maxf = maxf.data.cpu().numpy()
        avgf = avgf.data.cpu().numpy()

    model.eval()

    lay = 'g_layers'
    progress_bar = tqdm(data)
    progress_bar.set_description('FEATURES EXTRACTION from {}, position {}'.format(lay, quest_inject_index))
    max_features = []
    avg_features = []

    extraction_layer = model._modules.get('rl')._modules.get(lay)[quest_inject_index]
    h = extraction_layer.register_forward_hook(hook_function)
    for batch_idx, sample_batched in enumerate(progress_bar):
        qst = torch.LongTensor(len(sample_batched), 1).zero_()
        qst = Variable(qst)

        img = Variable(sample_batched)
        if args.cuda:
            qst = qst.cuda()
            img = img.cuda()
        
        model(img, qst)

        max_features.append((batch_idx, maxf))
        avg_features.append((batch_idx, avgf))

    h.remove()

    pickle.dump(max_features, max_features_file)
    pickle.dump(avg_features, avg_features_file)

def reload_loaders(clevr_dataset_test, test_bs, state_description = False): #TODO here: add custom collect function
    if not state_description:

        # Initialize Clevr dataset loader
        clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs,
                                       shuffle=False, num_workers=8)
    else:
        # Initialize Clevr dataset loader
        clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=test_bs,
                                       shuffle=False, num_workers=1, collate_fn=utils.collate_samples_images_state_description)
    return clevr_test_loader

def initialize_dataset(clevr_dir, state_description=True):
    if not state_description:
        test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])
                                          
        clevr_dataset_test = ClevrDatasetImages(clevr_dir, False, test_transforms)
        
    else:
        clevr_dataset_test = ClevrDatasetImagesStateDescription(clevr_dir, False)
    
    return clevr_dataset_test 


def main(args):
    #load hyperparameters from configuration file
    with open(args.config) as config_file: 
        hyp = json.load(config_file)['hyperparams'][args.model]
    #override configuration dropout
    if args.question_injection >= 0:
        hyp['question_injection_position'] = args.question_injection

    print('Loaded hyperparameters from configuration {}, model: {}: {}'.format(args.config, args.model, hyp))

    #assert os.path.isfile(args.checkpoint), "Checkpoint file not found: {}".format(args.checkpoint)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Initialize CLEVR Loader
    clevr_dataset_test  = initialize_dataset(args.clevr_dir, hyp['state_description'])
    clevr_feat_extraction_loader = reload_loaders(clevr_dataset_test, args.batch_size, hyp['state_description'])

    args.features_dirs = './features'
    if not os.path.exists(args.features_dirs):
        os.makedirs(args.features_dirs)

    max_features = os.path.join(args.features_dirs, 'max_features.pickle')
    avg_features = os.path.join(args.features_dirs, 'avg_features.pickle')

    args.qdict_size = 0
    args.adict_size = 0
    model = RN(args, hyp, extraction=True)

    if torch.cuda.device_count() > 1 and args.cuda:
        model = torch.nn.DataParallel(model)
        model.module.cuda()  # call cuda() overridden method

    if args.cuda:
        model.cuda()

    # Load the model checkpoint
    '''print('==> loading checkpoint {}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)

    #removes 'module' from dict entries, pytorch bug #3805
    checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    print('==> loaded checkpoint {}'.format(args.checkpoint))'''

    max_features = open(max_features, 'wb')
    avg_features = open(avg_features, 'wb')

    extract_features_rl(clevr_feat_extraction_loader, hyp['question_injection_position'], max_features, avg_features, model, args)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR Feature Extraction')
    parser.add_argument('--checkpoint', type=str,
                        help='model checkpoint to use for feature extraction')
    parser.add_argument('--model', type=str, default='original-fp',
                        help='which model is used to train the network')
    parser.add_argument('--layer', type=str, default='unused:o',
                        help='layer of the RN from which features are extracted')
    parser.add_argument('--clevr-dir', type=str, default='.',
                        help='base directory of CLEVR dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--config', type=str, default='config.json',
                        help='configuration file for hyperparameters loading')
    parser.add_argument('--question-injection', type=int, default=-1, 
                        help='At which stage of g function the question should be inserted (0 to insert at the beginning, as specified in DeepMind model, -1 to use configuration value)')
    args = parser.parse_args()
    main(args)
