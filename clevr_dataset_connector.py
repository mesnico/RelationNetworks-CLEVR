import os
import json
import cv2
import utils
from torch.utils.data import Dataset
import pdb

class ClevrDataset(Dataset):
    """Face Landmarks dataset."""

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

        json_file = open(json_filename)
        self.questions = json.load(json_file)['questions']
        self.clevr_dir = clevr_dir
        self.transform = transform
        self.dictionaries = dictionaries

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        img_filename = os.path.join(self.img_dir, current_question['image_filename'])
        image = cv2.imread(img_filename)
        
        question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])

        sample = {'image': image, 'question': question, 'answer': answer}

        if self.transform:
            sample = self.transform(sample)

        #pdb.set_trace()
        return sample
