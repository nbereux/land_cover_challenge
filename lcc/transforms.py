from torchvision import transforms
import numpy as np

import pdb

class Normalize(object):

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, sample):        
        sample['image'] = (sample['image'] - self.min) / (self.max - self.min)
        return sample

class AllTransforms(object):

    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.normalize = Normalize(self.min, self.max)
        self.to_tensor = transforms.ToTensor()
        self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        self.random_vertical_flip = transforms.RandomVerticalFlip(p=1)
        self.random_crop = transforms.RandomCrop(256)
        self.jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        self.rotation_1 = transforms.RandomRotation(90)
        self.rotation_2 = transforms.RandomRotation(270)
        

    def __call__(self, sample):
        sample = self.normalize(sample)

        image, mask = sample['image'], sample['mask']
        image = self.to_tensor(image).float()
        mask = self.to_tensor(mask).long()
        u = np.random.rand()
        if np.random.rand() < 0.5:
            image = self.random_horizontal_flip(image)
            mask = self.random_horizontal_flip(mask)
        if np.random.rand() < 0.5:
            image = self.random_vertical_flip(image)
            mask = self.random_vertical_flip(mask)
        # if np.random.rand() < 0.5:
        #     params = self.random_crop.get_params(image, (256, 256))
        #     image = image[params[0]:params[2], params[1]:params[3]]
        #     mask = mask[params[0]:params[2], params[1]:params[3]]
        if np.random.rand() < 0.5:
            image = self.rotation_1(image)
            mask = self.rotation_1(mask)
        if np.random.rand() < 0.5:
            image = self.rotation_2(image)
            mask = self.rotation_2(mask)
        #image[:3] = self.jitter(image[:3])
        sample['image'] = image
        sample['mask'] = mask
        return sample