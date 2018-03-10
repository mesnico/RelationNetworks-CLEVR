import argparse
import os
import json
import numpy as np
from PIL import Image
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import ConvInputModel
from torchvision import transforms

from sklearn.metrics import average_precision_score

from tqdm import tqdm, trange
import pdb

class ClevrDatasetForMulticlass(Dataset):
    def __init__(self, clevr_dir, train, perc, transform):
        attributes = ['material','color','shape','size']
        attr_values = ['rubber','metal', 'cyan','blue','yellow','purple','red','green','gray','brown','sphere','cube','cylinder','large','small']

        if train:
            json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_train_scenes.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'train')
        else:
            json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'val')
        
        # build up targets for all questions
        targets = []
        with open(json_filename, 'r') as json_file:
            scenes = json.load(json_file)['scenes']
        for scene in scenes:
            attr_onehot = np.zeros(len(attr_values))
            for obj in scene['objects']:
                for attr in obj: 
                    if attr in attributes:
                        idx = attr_values.index(obj[attr])
                        attr_onehot[idx] = 1
            targets.append((scene['image_filename'],attr_onehot))

        # filter targets by numbers of ones in the one-hot vectors
        targets = sorted(targets, key=lambda x: sum(x[1]))
        self.num = math.floor(len(scenes) * perc)
        targets = targets[0:self.num]
        self.img_filenames = [x[0] for x in targets]
        self.targets = [torch.from_numpy(x[1]).float() for x in targets]
        print('[{} - Last target vector is {}'.format('train' if train else 'test', self.targets[-1]))
        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_filename = os.path.join(self.img_dir, self.img_filenames[idx])
        image = Image.open(img_filename).convert('RGB')

        target = self.targets[idx]

        '''if self.dictionaries[2][answer[0]]=='color':
            image = Image.open(img_filename).convert('L')
            image = numpy.array(image)
            image = numpy.stack((image,)*3)
            image = numpy.transpose(image, (1,2,0))
            image = Image.fromarray(image.astype('uint8'), 'RGB')'''
        
        sample = {'image': image, 'target': target}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample

class MulticlassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvInputModel()
        self.fc1 = nn.Linear(24, 15)
        self.fc2 = nn.Linear(15, 15)

    def forward(self, img):
        x = self.conv(img)
        bs = x.size()[0]
        #global max pooling

        x = x.view(bs, 24, 8**2).sum(2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
        
def collate_samples(batch):
    """
    Used by DatasetLoader to merge together multiple samples into one mini-batch.
    """
    images = [d['image'] for d in batch]
    targets = [d['target'] for d in batch]

    collated_batch = dict(
        image=torch.stack(images),
        target=torch.stack(targets)
    )

    return collated_batch

def load_tensor_data(data_batch, cuda, volatile=False):
    # prepare input
    var_kwargs = dict(volatile=True) if volatile else dict(requires_grad=False)

    img = torch.autograd.Variable(data_batch['image'], **var_kwargs)
    target = torch.autograd.Variable(data_batch['target'], **var_kwargs)
    if cuda:
        img, target = img.cuda(), target.cuda()

    return img, target

def train(data, model, optimizer, epoch, args):
    model.train()
    loss_funct = nn.MultiLabelSoftMarginLoss()

    avg_loss = 0.0
    n_batches = 0
    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, target = load_tensor_data(sample_batched, args.cuda, volatile=True)

        # forward and backward pass
        optimizer.zero_grad()
        output = model(img)
        loss = loss_funct(output, target)
        loss.backward()

        optimizer.step()

        # Show progress
        progress_bar.set_postfix(dict(loss=loss.data[0]))
        avg_loss += loss.data[0]
        n_batches += 1

        if batch_idx % args.log_interval == 0:
            avg_loss /= n_batches
            processed = batch_idx * args.batch_size
            n_samples = len(data) * args.batch_size
            progress = float(processed) / n_samples
            print('Train Epoch: {} [{}/{} ({:.0%})] Train loss: {}'.format(
                epoch, processed, n_samples, progress, avg_loss))
            avg_loss = 0.0
            n_batches = 0

def test(data, model, epoch, args):
    model.eval()

    n_iters = 0
    ap_sum = 0.0

    progress_bar = tqdm(data)
    for batch_idx, sample_batched in enumerate(progress_bar):
        img, target = load_tensor_data(sample_batched, args.cuda, volatile=True)
        
        output = model(img)
        ap = average_precision_score(target, output.data) 
        n_iters += 1
        ap_sum += ap
        if batch_idx % args.log_interval == 0:
            m_ap = ap_sum / n_iters
            progress_bar.set_postfix(dict(AP='{:.2}'.format(m_ap)))

    m_ap = ap_sum / n_iters
    print('Test Epoch {}: Avg. Precision Score = {:.2};'.format(epoch, m_ap))

def main(args):
    args.model_dirs = './cnn_model_b{}_lr{}'.format(args.batch_size, args.lr)
    if not os.path.exists(args.model_dirs):
        os.makedirs(args.model_dirs)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('Initializing CLEVR dataset...')
    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.Pad(8),
                                           transforms.RandomCrop((128, 128)),
                                           transforms.RandomRotation(2.8),  # .05 rad
                                           transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    clevr_dataset_train = ClevrDatasetForMulticlass(args.clevr_dir, True, 0.05, train_transforms)
    clevr_dataset_test = ClevrDatasetForMulticlass(args.clevr_dir, False, 0.05, test_transforms)
    
    # Initialize Clevr dataset loaders
    clevr_train_loader = DataLoader(clevr_dataset_train, batch_size=args.batch_size,
                                    shuffle=True, num_workers=8, collate_fn=collate_samples)
    clevr_test_loader = DataLoader(clevr_dataset_test, batch_size=args.batch_size,
                                   shuffle=False, num_workers=8, collate_fn=collate_samples)

    print('CLEVR dataset initialized!')

    # Build the model
    model = MulticlassificationModel()

    #if torch.cuda.device_count() > 1 and args.cuda:
    #    model = torch.nn.DataParallel(model)
    #    model.module.cuda()  # call cuda() overridden method

    if args.cuda:
        model.cuda()

    start_epoch = 1
    if args.resume:
        filename = args.resume
        if os.path.isfile(filename):
            print('==> loading checkpoint {}'.format(filename))
            checkpoint = torch.load(filename)

            #removes 'module' from dict entries, pytorch bug #3805
            checkpoint = {k.replace('module.',''): v for k,v in checkpoint.items()}

            model.load_state_dict(checkpoint)
            print('==> loaded checkpoint {}'.format(filename))
            start_epoch = int(re.match(r'.*epoch_(\d+).pth', args.resume).groups()[0]) + 1

    progress_bar = trange(start_epoch, args.epochs + 1)
    if args.test:
        #perform a single test
        print('Testing epoch {}'.format(start_epoch))
        test(clevr_test_loader, model, start_epoch, args)
    else:
        #perform a full training
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print('Training ({} epochs) is starting...'.format(args.epochs))
        for epoch in progress_bar:
            # TRAIN
            progress_bar.set_description('TRAIN')
            train(clevr_train_loader, model, optimizer, epoch, args)
            # TEST
            progress_bar.set_description('TEST')
            test(clevr_test_loader, model, epoch, args)
            # SAVE MODEL
            filename = 'RN_epoch_{:02d}.pth'.format(epoch)
            torch.save(model.state_dict(), os.path.join(args.model_dirs, filename))

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
    parser.add_argument('--test', action='store_true', default=False,
                        help='perform only a single test. To use with --resume')

    args = parser.parse_args()
    main(args)
