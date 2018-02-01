from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.optim as optim
import torch
import os

def start(start_epoch, clevr_train_loader, clevr_test_loader, model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print('Training ({} epochs) is starting...'.format(args.epochs))
    progress_bar = trange(start_epoch, args.epochs + 1)
    for epoch in progress_bar:
        # TRAIN
        progress_bar.set_description('TRAIN')
        train(clevr_train_loader, model, optimizer, epoch, args)
        # TEST
        progress_bar.set_description('TEST')
        test(clevr_test_loader, model, epoch, args)
        # SAVE MODEL
        fname = 'RN_epoch_{:02d}.pth'.format(epoch)
        torch.save(model.state_dict(), os.path.join(args.model_dirs, fname))

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

        # Gradient Clipping
        clip_grad_norm(model.parameters(), 10)
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
