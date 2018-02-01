import re

import torch


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
