import torch
import re

'''
    Outputs indexes of the dictionary corresponding to the words in the sequence. Case insensitive
'''
def to_dictionary_indexes(dictionary, sentence):
    split = tokenize(sentence)
    idxs = torch.LongTensor([dictionary[w] for w in split])
    return idxs

'''
    Used by DatasetLoader to merge together multiple samples into one mini-batch
'''
def collate_samples(batch):
    images = [d['image'] for d in batch]
    answers = [d['answer'] for d in batch]
    questions = [d['question'] for d in batch]
    
    # questions are not fixed length: they must be padded to the maximum length 
    # in this batch, in order to be inserted in a tensor
    batch_size = len(batch)
    max_len = max(map(len, questions))
    
    padded_questions = torch.LongTensor(batch_size, max_len).zero_()
    for i, q in enumerate(questions):
        padded_questions[i,:len(q)] = q

    #invert question indexes
    padded_questions = padded_questions.index_select(1,torch.arange(max_len-1, -1, -1).long())
    
    collated_batch = dict(
        image=torch.stack(images),
        answer=torch.stack(answers),
        question=torch.stack(padded_questions)
    )
    
    return collated_batch

def tokenize(sentence):
    #punctuation should be separated from the words
    s = re.sub('([.,;:!?()])', r' \1 ', sentence)
    s = re.sub('\s{2,}', ' ', s)

    #tokenize
    split = s.split()

    #normalize all words to lowercase
    lower = [w.lower() for w in split]
    return lower

