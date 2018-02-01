"""
Pytorch implementation of "A simple neural network module for relational reasoning"
"""
from __future__ import print_function

import argparse
import os
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import utils
from clevr_dataset_connector import ClevrDatasetImages
from model import RN


def extract_features_rl(data, max_features_file, avg_features_file, model, args):
    assert args.layer, "A layer must be specified."
    lay, io = args.layer.split(':')

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

    progress_bar = tqdm(data)
    progress_bar.set_description('FEATURES EXTRACTION from {}'.format(args.layer))
    max_features = []
    avg_features = []
    for batch_idx, sample_batched in enumerate(progress_bar):
        qst = torch.LongTensor(args.batch_size, 1).zero_()
        qst = Variable(qst)

        img = Variable(sample_batched)
        if args.cuda:
            qst = qst.cuda()
            img = img.cuda()

        extraction_layer = model._modules.get('rl')._modules.get(lay)
        h = extraction_layer.register_forward_hook(hook_function)
        model(img, qst)
        h.remove()

        max_features.append((batch_idx, maxf))
        avg_features.append((batch_idx, maxf))

    pickle.dump(max_features, max_features_file)
    pickle.dump(avg_features, avg_features_file)


def main(args):
    assert os.path.isfile(args.checkpoint), "Checkpoint file not found: {}".format(args.checkpoint)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    # Initialize CLEVR Loader
    clevr_dataset_images = ClevrDatasetImages(args.clevr_dir, 'val', test_transforms)
    clevr_feat_extraction_loader = DataLoader(clevr_dataset_images, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8, drop_last=True)

    args.features_dirs = './features'
    if not os.path.exists(args.features_dirs):
        os.makedirs(args.features_dirs)

    max_features = os.path.join(args.features_dirs, 'max_features.pickle')
    avg_features = os.path.join(args.features_dirs, 'avg_features.pickle')

    print('Building word dictionaries from all the words in the dataset...')
    dictionaries = utils.build_dictionaries(args.clevr_dir)
    print('Word dictionary completed!')

    args.qdict_size = len(dictionaries[0])
    args.adict_size = len(dictionaries[1])
    model = RN(args)

    if torch.cuda.device_count() > 1 and args.cuda:
        model = torch.nn.DataParallel(model)
        model.module.cuda()  # call cuda() overridden method

    if args.cuda:
        model.cuda()

    # Load the model checkpoint
    print('==> loading checkpoint {}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    print('==> loaded checkpoint {}'.format(args.checkpoint))

    max_features = open(max_features, 'wb')
    avg_features = open(avg_features, 'wb')

    extract_features_rl(clevr_feat_extraction_loader, max_features, avg_features, model, args)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Relational-Network CLEVR Feature Extraction')
    parser.add_argument('checkpoint', type=str,
                        help='model checkpoint to use for feature extraction')
    parser.add_argument('--model', type=str, choices=['original', 'ir'], default='original',
                        help='which model is used to train the network')
    parser.add_argument('--layer', type=str, default=None,  # TODO set a default layer
                        help='layer of the RN from which features are extracted')
    parser.add_argument('--clevr-dir', type=str, default='.',
                        help='base directory of CLEVR dataset')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    main(args)
