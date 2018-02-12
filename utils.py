import json
import os
import pickle
import re

import torch
from tqdm import tqdm
import random

def build_dictionaries(clevr_dir):

    def compute_class(answer):
        classes = {
            'number':['0','1','2','3','4','5','6','7','8','9','10'],
            'material':['rubber','metal'],
            'color':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape':['sphere','cube','cylinder'],
            'size':['large','small'],
            'exist':['yes','no']
        }

        for name,values in classes.items():
            if answer in values:
                return name
        
        raise ValueError('Answer {} does not belong to a known class'.format(answer))
        
        
    cached_dictionaries = os.path.join(clevr_dir, 'questions', 'CLEVR_built_dictionaries.pkl')
    if os.path.exists(cached_dictionaries):
        print('==> using cached dictionaries: {}'.format(cached_dictionaries))
        with open(cached_dictionaries, 'rb') as f:
            return pickle.load(f)
            
    quest_to_ix = {}
    answ_to_ix = {}
    answ_ix_to_class = {}
    json_train_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
    #load all words from all training data
    with open(json_train_filename, "r") as f:
        questions = json.load(f)['questions']
        #questions = [s for s in questions if compute_class(s['answer']) == 'exist']
        for q in tqdm(questions):
            question = tokenize(q['question'])
            answer = q['answer']
            #pdb.set_trace()
            for word in question:
                if word not in quest_to_ix:
                    quest_to_ix[word] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding
            
            a = answer.lower()
            if a not in answ_to_ix:
                    ix = len(answ_to_ix)+1
                    answ_to_ix[a] = ix
                    answ_ix_to_class[ix] = compute_class(a)

    ret = (quest_to_ix, answ_to_ix, answ_ix_to_class)    
    with open(cached_dictionaries, 'wb') as f:
        pickle.dump(ret, f)

    return ret


def to_dictionary_indexes(dictionary, sentence):
    """
    Outputs indexes of the dictionary corresponding to the words in the sequence.
    Case insensitive.
    """
    split = tokenize(sentence)
    idxs = torch.LongTensor([dictionary[w] for w in split])
    return idxs


def collate_samples(batch):
    """
    Used by DatasetLoader to merge together multiple samples into one mini-batch.
    """
    images = [d['image'] for d in batch]
    answers = [d['answer'] for d in batch]
    questions = [d['question'] for d in batch]

    # questions are not fixed length: they must be padded to the maximum length
    # in this batch, in order to be inserted in a tensor
    batch_size = len(batch)
    max_len = max(map(len, questions))

    padded_questions = torch.LongTensor(batch_size, max_len).zero_()
    for i, q in enumerate(questions):
        padded_questions[i, :len(q)] = q

    collated_batch = dict(
        image=torch.stack(images),
        answer=torch.stack(answers),
        question=torch.stack(padded_questions)
    )

    return collated_batch


def tokenize(sentence):
    # punctuation should be separated from the words
    s = re.sub('([.,;:!?()])', r' \1 ', sentence)
    s = re.sub('\s{2,}', ' ', s)

    # tokenize
    split = s.split()

    # normalize all words to lowercase
    lower = [w.lower() for w in split]
    return lower


def load_tensor_data(data_batch, cuda, invert_questions, volatile=False):
    # prepare input
    var_kwargs = dict(volatile=True) if volatile else dict(requires_grad=False)

    qst = data_batch['question']
    if invert_questions:
        # invert question indexes in this batch
        qst_len = qst.size()[1]
        qst = qst.index_select(1, torch.arange(qst_len - 1, -1, -1).long())

    img = torch.autograd.Variable(data_batch['image'], **var_kwargs)
    qst = torch.autograd.Variable(qst, **var_kwargs)
    label = torch.autograd.Variable(data_batch['answer'], **var_kwargs)
    if cuda:
        img, qst, label = img.cuda(), qst.cuda(), label.cuda()

    label = (label - 1).squeeze(1)
    return img, qst, label

class ClevrClassSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, clevr_dir, dictionaries, bs):
        self.bs = bs
        self.dictionaries = dictionaries
        json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
        with open(json_filename, 'r') as json_file:
            questions = json.load(json_file)['questions']

        self.indexes_per_class = []
        self.classes = set(dictionaries[2].values())
        for c in self.classes:
            filtered = [idx for idx,s in enumerate(questions) if dictionaries[2][dictionaries[1][s['answer']]] == c]
            self.indexes_per_class.append(filtered)
        self.counts = [len(idxs) for idxs in self.indexes_per_class]
        
    def __iter__(self):
        perms = [random.sample(idxs, len(idxs)) for idxs in self.indexes_per_class]
        counts = list(self.counts)
        classes_idx = list(range(len(self.classes)))
        rand_class = random.choice(classes_idx)
        counts = list(self.counts)
        batch = []
        while len(classes_idx) != 0:
            #pdb.set_trace()
            idx = perms[rand_class][counts[rand_class]-1]
            batch.append(idx)
            counts[rand_class] -= 1
            if len(batch) == self.bs:
                yield batch
                if counts[rand_class] < self.bs:
                    del classes_idx[classes_idx.index(rand_class)]
                if len(classes_idx) != 0:
                    rand_class = random.choice(classes_idx)
                batch = []

    def __len__(self):
        total = 0
        for c in self.counts:
            total += c // self.bs
        return total
