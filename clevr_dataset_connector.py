import os
import json
import utils
import torch

from PIL import Image
from torch.utils.data import Dataset

class ClevrDataset(Dataset):

    def __init__(self, clevr_dir, train, dictionaries, transform=None):
        """
        Args:
            clevr_dir (string): Root directory of CLEVR dataset 
			train (bool): Tells if we are loading the train or the validation datasets
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'train')
        else:
            json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_val_questions.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'val')

        with open(json_filename, 'r') as json_file:
            self.questions = json.load(json_file)['questions']
        self.clevr_dir = clevr_dir
        self.transform = transform
        self.dictionaries = dictionaries

        #calculate longest question length
        qs_len = [len(e['question'].split()) for e in self.questions]   #TODO: punctuation?
        self.max_qlength = max(qs_len)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        img_filename = os.path.join(self.img_dir, current_question['image_filename'])
        image = Image.open(img_filename).convert('RGB')
        
        pad_question = torch.LongTensor(self.max_qlength).zero_()
        question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        pad_question[:len(question)] = question
        
        answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])

        sample = {'image': image, 'question': pad_question, 'answer': answer}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
