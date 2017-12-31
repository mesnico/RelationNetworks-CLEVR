import torch

'''
    Outputs indexes of the dictionary corresponding to the words in the sequence. Case insensitive
'''
def to_dictionary_indexes(dictionary, sentence):
    split = sentence.split()	#TODO: punctuation?
    idxs = torch.LongTensor([dictionary[w.lower()] for w in split])
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
    '''batch_size = len(batch)
    max_len = max(map(len, questions))
    
    padded_questions = torch.LongTensor(batch_size, max_len).zero_()
    for i, q in enumerate(questions):
        padded_questions[i,:len(q)] = q'''
    
    collated_batch = dict(
        image=torch.stack(images),
        answer=torch.stack(answers),
        question=torch.stack(questions)
    )
    
    return collated_batch

