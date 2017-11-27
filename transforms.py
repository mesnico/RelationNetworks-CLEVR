import torch
import cv2
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, qst, ans = sample['image'], sample['question'], sample['answer']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        img = img / 255.
        return {'image': torch.from_numpy(img),
                'question': torch.from_numpy(qst),
                'answer': torch.from_numpy(ans)}

class Resize(object):
    """Resize the image to the specified dimensions."""
    def __init__(self, output_size):
        assert len(output_size)==2
        self.output_size = output_size
    
    def __call__(self, sample):
        img, qst, ans = sample['image'], sample['question'], sample['answer']

        img = cv2.resize(img, self.output_size)
        return {'image': img,
                'question': qst,
                'answer': ans}

class Pad(object):
    """Pad the image of a certain amount"""
    def __init__(self, pad_amount):
        assert isinstance(pad_amount, (int))
        self.pad_amount = pad_amount
    
    def __call__(self, sample):
        img, qst, ans = sample['image'], sample['question'], sample['answer']

        img = cv2.copyMakeBorder(img,self.pad_amount,self.pad_amount,self.pad_amount,self.pad_amount,
            cv2.BORDER_CONSTANT,value=[255,255,255])
        return {'image': img,
                'question': qst,
                'answer': ans}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img, qst, ans = sample['image'], sample['question'], sample['answer']

        h, w = img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h,
                      left: left + new_w]

        return {'image': img,
                'question': qst,
                'answer': ans}

class RandomRotate(object):
    """Resize the image to the specified dimensions."""
    def __init__(self, degrees):
        assert isinstance(degrees, (int, float))
        self.degrees = degrees
    
    def __call__(self, sample):
        img, qst, ans = sample['image'], sample['question'], sample['answer']

        rows,cols = img.shape[:2]
        random_angle = (np.random.random() * 2 - 1) * self.degrees

        M = cv2.getRotationMatrix2D((cols/2,rows/2),random_angle,1)
        img = cv2.warpAffine(img,M,(cols,rows),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return {'image': img,
                'question': qst,
                'answer': ans}



