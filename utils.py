import json
import os
import pickle
import re
import shelve

import torch
from tqdm import tqdm

classes = {
            'number':['0','1','2','3','4','5','6','7','8','9','10'],
            'material':['rubber','metal'],
            'color':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape':['sphere','cube','cylinder'],
            'size':['large','small'],
            'exist':['yes','no']
        }

def build_dictionaries(clevr_dir):

    def compute_class(answer):
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


def to_dictionary_indexes(dictionary, sentence, invert):
    """
    Outputs indexes of the dictionary corresponding to the words in the sequence.
    Case insensitive.
    """
    split = tokenize(sentence)
    d = [dictionary[w] for w in split]
    if invert:
        d = d[::-1]
    idxs = torch.LongTensor(d)
    return idxs

def collate_samples_from_pixels(batch):
    return collate_samples(batch, False, False)
    
def collate_samples_state_description(batch):
    return collate_samples(batch, True, False)

def collate_samples_images_state_description(batch):
    return collate_samples(batch, True, True)
    
def collate_samples(batch, state_description, only_images):
    """
    Used by DatasetLoader to merge together multiple samples into one mini-batch.
    """
    batch_size = len(batch)

    if only_images:
        images = batch
    else:
        images = [d['image'] for d in batch]
        answers = [d['answer'] for d in batch]
        questions = [d['question'] for d in batch]

        # questions are not fixed length: they must be padded to the maximum length
        # in this batch, in order to be inserted in a tensor
        max_len = max(map(len, questions))

        padded_questions = torch.LongTensor(batch_size, max_len).zero_()
        question_lengths = torch.LongTensor(batch_size).zero_()
        for i, q in enumerate(questions):
            padded_questions[i, :len(q)] = q
            question_lengths[i] = len(q)
        
    if state_description:
        max_len = 12
        #even object matrices should be padded (they are variable length)
        padded_objects = torch.FloatTensor(batch_size, max_len, images[0].size()[1]).zero_()
        for i, o in enumerate(images):
            padded_objects[i, :o.size()[0], :] = o
        images = padded_objects
    
    if only_images:
        collated_batch = torch.stack(list(images))
    else:
        # permute batch so that questions are sorted by increasing order (for packed sequence)
        images = torch.stack(list(images))
        answers = torch.stack(list(answers))

        question_lengths, idxs = question_lengths.sort(descending=True)
        images = images.index_select(0, idxs)
        answers = answers.index_select(0, idxs)
        padded_questions = padded_questions.index_select(0, idxs)       

        collated_batch = dict(
            image=images,
            answer=answers,
            question=padded_questions,
            lengths=question_lengths
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


def load_tensor_data(data_batch, cuda, volatile=False):
    # prepare input
    var_kwargs = dict(volatile=True) if volatile else dict(requires_grad=False)

    qst = data_batch['question']
    #qst_lenghts = data_batch['question_lenghts']
    '''if invert_questions:
        # invert question indexes in this batch
        qst_len = qst.size()[1]
        qst = qst.index_select(1, torch.arange(qst_len - 1, -1, -1).long())
    '''
    img = torch.autograd.Variable(data_batch['image'], **var_kwargs)
    qst = torch.autograd.Variable(qst, **var_kwargs)
    label = torch.autograd.Variable(data_batch['answer'], **var_kwargs)
    qst_len = torch.autograd.Variable(data_batch['lengths'], **var_kwargs)
    if cuda:
        img, qst, label, qst_len = img.cuda(), qst.cuda(), label.cuda(), qst_len.cuda()

    label = (label - 1).squeeze(1)
    return img, qst, label, qst_len

class JsonCache:
    def __init__(self, json_filename, what, all_in_memory=True, json_cook_function=None):
        self.all_in_memory = all_in_memory
        cached_filename = json_filename.replace('.json', '.pkl' if all_in_memory else '.db')
        if os.path.exists(cached_filename) or os.path.exists(cached_filename+'.dat'):
            print('==> using cached: {}'.format(cached_filename))
            if all_in_memory:
                with open(cached_filename, 'rb') as f:
                    self.out = pickle.load(f)
            else:
                self.out = shelve.open(cached_filename)
                
        else:
            with open(json_filename, 'r') as json_file:
                loaded = json.load(json_file)[what]
                if json_cook_function != None:
                    loaded = json_cook_function(loaded)
            if all_in_memory:
                with open(cached_filename, 'wb') as f:
                    pickle.dump(loaded, f)
                    self.out = loaded
            else:
                with shelve.open(cached_filename) as s:
                    for idx, x in enumerate(loaded):
                        s[str(idx)] = x
                self.out = shelve.open(cached_filename)

    def __getitem__(self,idx):
        if self.all_in_memory:
            return self.out[idx]
        else:
            return self.out[str(idx)]

    def __len__(self):
        return len(self.out)
        
