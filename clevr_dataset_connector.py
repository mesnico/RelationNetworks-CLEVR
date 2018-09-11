import json
import os
import pickle
from PIL import Image

from collections import Counter
from torch.utils.data import Dataset

import utils
import torch

class ClevrDataset(Dataset):
    def __init__(self, clevr_dir, train, dictionaries, transform=None, all_in_memory=False):
        """
        Args:
            clevr_dir (string): Root directory of CLEVR dataset
			train (bool): Tells if we are loading the train or the validation datasets
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if train:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'train')
        else:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_val_questions.json')
            self.img_dir = os.path.join(clevr_dir, 'images', 'val')

        self.questions = utils.JsonCache(quest_json_filename, 'questions', all_in_memory)
                
        self.clevr_dir = clevr_dir
        self.transform = transform
        self.dictionaries = dictionaries
    
    def answer_weights(self):
        n = float(len(self.questions))
        answer_count = Counter(q['answer'].lower() for q in self.questions)
        weights = [n/answer_count[q['answer'].lower()] for q in self.questions]
        return weights
    
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        img_filename = os.path.join(self.img_dir, current_question['image_filename'])
        image = Image.open(img_filename).convert('RGB')

        question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])
        '''if self.dictionaries[2][answer[0]]=='color':
            image = Image.open(img_filename).convert('L')
            image = numpy.array(image)
            image = numpy.stack((image,)*3)
            image = numpy.transpose(image, (1,2,0))
            image = Image.fromarray(image.astype('uint8'), 'RGB')'''
        
        sample = {'image': image, 'question': question, 'answer': answer}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample                

class ClevrDatasetStateDescription(Dataset):
    def __init__(self, clevr_dir, train, dictionaries, all_in_memory=False):
        
        if train:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')
            scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_train_scenes.json')
        else:
            quest_json_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_val_questions.json')
            scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_val_scenes.json')

        def get_objects(scenes):
            all_scene_objs = []
            print('caching all objects in all scenes...')
            for s in scenes:
                objects = s['objects']
                objects_attr = []
                for obj in objects:
                    attr_values = []
                    for attr in sorted(obj):
                        # convert object attributes in indexes
                        if attr in utils.classes:
                            attr_values.append(utils.classes[attr].index(obj[attr])+1)  #zero is reserved for padding
                        else:
                            '''if attr=='rotation':
                                attr_values.append(float(obj[attr]) / 360)'''
                            if attr=='3d_coords':
                                attr_values.extend(obj[attr])
                    objects_attr.append(attr_values)
                all_scene_objs.append(torch.FloatTensor(objects_attr))
            return all_scene_objs

        self.questions = utils.JsonCache(quest_json_filename, 'questions', all_in_memory)
        self.objects = utils.JsonCache(scene_json_filename, 'scenes', all_in_memory, json_cook_function=get_objects)             
                
        self.clevr_dir = clevr_dir
        self.dictionaries = dictionaries
    
    '''def answer_weights(self):
        n = float(len(self.questions))
        answer_count = Counter(q['answer'].lower() for q in self.questions)
        weights = [n/answer_count[q['answer'].lower()] for q in self.questions]
        return weights'''
    
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        scene_idx = current_question['image_index']
        obj = self.objects[scene_idx]
        
        
        question = utils.to_dictionary_indexes(self.dictionaries[0], current_question['question'])
        answer = utils.to_dictionary_indexes(self.dictionaries[1], current_question['answer'])
        '''if self.dictionaries[2][answer[0]]=='color':
            image = Image.open(img_filename).convert('L')
            image = numpy.array(image)
            image = numpy.stack((image,)*3)
            image = numpy.transpose(image, (1,2,0))
            image = Image.fromarray(image.astype('uint8'), 'RGB')'''
        
        sample = {'image': obj, 'question': question, 'answer': answer}
        
        return sample

class ClevrDatasetImages(Dataset):
    """
    Loads only images from the CLEVR dataset
    """

    def __init__(self, clevr_dir, train, transform=None):
        """
        :param clevr_dir: Root directory of CLEVR dataset
        :param mode: Specifies if we want to read in val, train or test folder
        :param transform: Optional transform to be applied on a sample.
        """
        self.mode = 'train' if train else 'val'
        self.img_dir = os.path.join(clevr_dir, 'images', self.mode)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        padded_index = str(idx).rjust(6, '0')
        img_filename = os.path.join(self.img_dir, 'CLEVR_{}_{}.png'.format(self.mode,padded_index))
        image = Image.open(img_filename).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class ClevrDatasetImagesStateDescription(ClevrDatasetStateDescription):
    def __init__(self, clevr_dir, train):
        super().__init__(clevr_dir, train, None)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        return self.objects[idx]    
