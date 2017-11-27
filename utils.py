import numpy as np
import torch
import pdb

'''
    Outputs indexes of the dictionary corresponding to the words in the sequence. Case insensitive
'''
def to_dictionary_indexes(dictionary, sentence):
    split = sentence.split()
    idxs = np.asarray([dictionary[w.lower()] for w in split])
    return idxs

'''
    Used by DatasetLoader to merge together multiple samples into one mini-batch
'''
def collate_samples(batch):
    l={}
    for key in batch[0]:
        #pdb.set_trace()
        if(key != 'question'):
            l[key] = torch.stack([d[key] for d in batch])
        elif(key == 'question'):
            #questions are not fixed length: they must be padded to the maximum length 
            #in this batch, in order to be inserted in a tensor

            #pdb.set_trace()
            varlen_list = [d[key] for d in batch]
            max_len = max([len(x) for x in varlen_list])
            l[key] = torch.stack([torch.cat((d[key], torch.LongTensor(max_len - len(d[key])).zero_()),0) for d in batch])
    return l
        
