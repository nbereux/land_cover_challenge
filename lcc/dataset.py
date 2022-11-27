import numpy as np
import os
from pathlib import Path
from tifffile import TiffFile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from lcc import DATASET_DIR
from lcc.transforms import Normalize

# image size of the images and label masks
IMG_SIZE = 256
# the images are RGB+NIR (4 channels)
N_CHANNELS = 4
# we have 9 classes + a 'no_data' class for pixels with no labels (absent in the dataset)
N_CLASSES = 10
CLASSES = [
    'no_data',
    'clouds',
    'artificial',
    'cultivated',
    'broadleaf',
    'coniferous',
    'herbaceous',
    'natural',
    'snow',
    'water'
]
# classes to ignore because they are not relevant. "no_data" refers to pixels without
# a proper class, but it is absent in the dataset; "clouds" class is not relevant, it
# is not a proper land cover type and images and masks do not exactly match in time.
IGNORED_CLASSES_IDX = [0, 1]

# The training dataset contains 18491 images and masks
# The test dataset contains 5043 images and masks
TRAINSET_SIZE = 18491
TESTSET_SIZE = 5043

# for visualization of the masks: classes indices and RGB colors
CLASSES_COLORPALETTE = {
    0: [0,0,0],
    1: [255,25,236],
    2: [215,25,28],
    3: [211,154,92],
    4: [33,115,55],
    5: [21,75,35],
    6: [118,209,93],
    7: [130,130,130],
    8: [255,255,255],
    9: [43,61,255]
    }
CLASSES_COLORPALETTE = {c: np.asarray(color) for (c, color) in CLASSES_COLORPALETTE.items()}

# statistics
# the pixel class counts in the training set
TRAIN_CLASS_COUNTS = np.array(
    [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605]
)
# the minimum and maximum value of image pixels in the training set
TRAIN_PIXELS_MIN = 1
TRAIN_PIXELS_MAX = 24356


class LCCDataset(Dataset):

    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.train_images_csv = np.genfromtxt(DATASET_DIR.joinpath('train_images.csv'), skip_header=True, dtype=np.int16)
        self.train_labels_csv = np.genfromtxt(DATASET_DIR.joinpath('train_labels.csv'), skip_header=True, dtype=np.int16)
        self.test_images_csv = np.genfromtxt(DATASET_DIR.joinpath('test_images.csv'), skip_header=True, dtype=np.int16)
        train_img_path = DATASET_DIR.joinpath('train', 'images')
        train_mask_path = DATASET_DIR.joinpath('train', 'masks')
        test_img_path = DATASET_DIR.joinpath('test', 'images')
        if not(os.path.exists(train_img_path)):
            raise FileNotFoundError(f"Training images not found at {train_img_path}")
        if not(os.path.exists(train_mask_path)):
            raise FileNotFoundError(f"Training masks not found at {train_mask_path}")
        if not(os.path.exists(test_img_path)):
            raise FileNotFoundError(f"Test images not found at {test_img_path}")
        self.train_img_path = train_img_path
        self.train_mask_path = train_mask_path
        self.test_img_path = test_img_path
        self.train = train

    def __len__(self):
        if self.train:
            return TRAINSET_SIZE
        else:
            return TESTSET_SIZE

    def __repr__(self) -> str:

        string = f"{self.__class__.__name__}(\n"
        string += f"    train={self.train},\n"
        string += f"    len={self.__len__()}\n"
        string += f")"
        return string

    def __getitem__(self, index):
        if index>=self.__len__():
            raise IndexError(f"Index {index} out of range [0, {self.__len__()}]")
        if self.train:
            img_path = self.train_img_path.joinpath(f'{self.train_images_csv[index]}.tif')
            mask_path = self.train_mask_path.joinpath(f'{self.train_images_csv[index]}.tif')
        else:
            img_path = self.test_img_path.joinpath(f'{self.test_images_csv[index]}.tif')
            mask_path = None
        mask = None
        with TiffFile(img_path) as tif:
            img = tif.asarray()
            if mask_path is not None:
                with TiffFile(mask_path) as tif:
                    mask = tif.asarray()
                    mask = mask[..., None]
        sample = {'image': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        sample['image'] = torch.from_numpy(sample['image']).to(torch.float32)
        sample['mask'] = torch.from_numpy(sample['mask']).to(torch.long)    
        return sample

def get_transforms(train=True):
    """Return a list of transformations to apply to the images and masks"""
    if train:
        return transforms.Compose([
            Normalize(1, TRAIN_PIXELS_MAX),
        ])