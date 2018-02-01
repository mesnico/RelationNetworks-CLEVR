from tqdm import tqdm
from torch.autograd import Variable
import pickle

def start(clevr_feat_extraction_loader, model, args):
    if not os.path.exists(args.features_dirs):
        os.makedirs(args.features_dirs)
    
    max_features = os.path.join(args.features_dirs,'max_features.pickle')
    avg_features = os.path.join(args.features_dirs,'avg_features.pickle')

    max_features = open(max_features, 'wb')
    avg_features = open(avg_features, 'wb')

    extract_features_rl(clevr_feat_extraction_loader,max_features,avg_features, model, args)

def extract_features_rl(data, max_features_file, avg_features_file, model, args):

    lay, io = args.extract_features.split(':')

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
            if io=='i':
                z = i[0]
            else:
                z = o
            #aggregate features
            d4_combinations = z.size()[0] // args.batch_size
            x_ = z.view(args.batch_size,d4_combinations,z.size()[1])
            maxf = x_.max(1)[0].squeeze()
            avgf = x_.mean(1).squeeze()
            
            maxf = maxf.data.cpu().numpy()
            avgf = avgf.data.cpu().numpy()

    model.eval()    

    progress_bar = tqdm(data)
    progress_bar.set_description('FEATURES EXTRACTION from {}'.format(args.extract_features))
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

        max_features.append((batch_idx,maxf))
        avg_features.append((batch_idx,maxf))
    
    pickle.dump(max_features, max_features_file)
    pickle.dump(avg_features, avg_features_file)

