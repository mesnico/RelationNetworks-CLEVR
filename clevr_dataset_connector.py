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

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        img_filename = os.path.join(self.img_dir, current_question['image_filename'])
        image = Image.open(img_filename).convert('RGB')
        
        question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'])       
        answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])

        sample = {'image': image, 'question': question, 'answer': answer}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


"""
    Loads only images from the CLEVR dataset
"""
class ClevrDatasetImages(Dataset):

    def __init__(self, clevr_dir, mode, transform=None):
        """
        Args:
            clevr_dir (string): Root directory of CLEVR dataset 
			mode (string): specifies if we want to read in val, train or test folder
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = os.path.join(clevr_dir, 'images', mode)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        padded_index = str(idx).rjust(6,'0')
        img_filename = os.path.join(self.img_dir, 'CLEVR_val_{}.png'.format(padded_index))
        image = Image.open(img_filename).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

