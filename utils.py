import json
import os
import pickle
import re
import shelve
import dgl
import networkx as nx
import numpy as np

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
    return collate_samples(batch, state_description=None, only_images=False)


def collate_samples_state_description_matrix(batch):
    return collate_samples(batch, state_description='matrix', only_images=False)


def collate_samples_state_description_graph(batch):
    return collate_samples(batch, state_description='graph', only_images=False)


def collate_samples_images_state_description(batch):
    return collate_samples(batch, state_description='matrix', only_images=True)

    
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
        
    if state_description == 'matrix':
        max_len = 12
        # even object matrices should be padded (they are variable length)
        padded_objects = torch.FloatTensor(batch_size, max_len, images[0].size()[1]).zero_()
        for i, o in enumerate(images):
            padded_objects[i, :o.size()[0], :] = o
        images = padded_objects
    
    if only_images:
        collated_batch = dgl.batch(images) if state_description == 'graph' else torch.stack(list(images))
    else:
        # permute batch so that questions are sorted by increasing order (for packed sequence)
        # images = torch.stack(list(images))
        answers = torch.stack(list(answers))

        question_lengths, idxs = question_lengths.sort(descending=True)
        images = [images[idx] for idx in idxs]
        # images = images.index_select(0, idxs)
        images = dgl.batch(images) if state_description == 'graph' else torch.stack(list(images))
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


def load_tensor_data(data_batch, cuda): #, volatile=False):
    # prepare input
    #var_kwargs = dict(volatile=True) if volatile else dict(requires_grad=False)

    qst = data_batch['question']
    #qst_lenghts = data_batch['question_lenghts']
    '''if invert_questions:
        # invert question indexes in this batch
        qst_len = qst.size()[1]
        qst = qst.index_select(1, torch.arange(qst_len - 1, -1, -1).long())
    '''
    img = data_batch['image']
    label = data_batch['answer']
    qst_len = data_batch['lengths']
    if cuda:
        qst, label, qst_len = qst.cuda(), label.cuda(), qst_len.cuda()
        if type(img) is dgl.BatchedDGLGraph:
            img.ndata['h'] = img.ndata['h'].cuda()
            img.edata['rel_type'] = img.edata['rel_type'].cuda()
        else:
            img = img.cuda()

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
                if json_cook_function is not None:
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

    def __getitem__(self, idx):
        if self.all_in_memory:
            return self.out[idx]
        else:
            return self.out[str(idx)]

    def __len__(self):
        return len(self.out)


def load_graphs(clevr_scenes):
    clevr_node_attrs = {
        'material': ['rubber', 'metal'],
        'color': ['cyan', 'blue', 'yellow', 'purple', 'red', 'green', 'gray', 'brown'],
        'shape': ['sphere', 'cube', 'cylinder'],
        'size': ['large', 'small']
    }

    clevr_relations_attrs = ['front', 'right', 'behind', 'left']

    '''with open(scene_json_filename, 'r') as jsonf:
        clevr_scenes = json.load(jsonf)['scenes']  # [0:1000]   # TODO: remove this, is only for debugging
    '''
    graphs = [0]*len(clevr_scenes)

    for scene in clevr_scenes:
        graph = nx.MultiDiGraph()
        # build graph nodes for every object
        objs = scene['objects']
        for idx, obj in enumerate(objs):
            # construct 15-dimensional one-hot node feature vector
            l = []
            for property in clevr_node_attrs:
                onehot = np.zeros(len(clevr_node_attrs[property]))
                onehot[clevr_node_attrs[property].index(obj[property])] = 1
                l.append(onehot)
            node_feat = torch.from_numpy(np.hstack(l)).float()
            graph.add_node(idx,
                           color=obj['color'],
                           shape=obj['shape'],
                           material=obj['material'],
                           size=obj['size'],
                           h=node_feat
                           )

        relationships = scene['relationships']
        for name, rel in relationships.items():
            # if name in ('right', 'front'):
                for b_idx, row in enumerate(rel):
                    for a_idx in row:
                        rel_id = clevr_relations_attrs.index(name)
                        rel_id = torch.tensor(rel_id)
                        graph.add_edge(a_idx, b_idx, rel_name=name, rel_type=rel_id)

        graphs[scene['image_index']] = graph
    return graphs
