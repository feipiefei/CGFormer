import torch.utils.data as data
import torch
from PIL import Image
from PIL import ImageFile
import os
import os.path
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import torch
import itertools
import random
from io import BytesIO
import torchvision.transforms as transforms

import time
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.TIF',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# using sort(): {'CGG': 0, 'Real': 1}
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort(key=len)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    num_in_class = []  # the number of samples in each class
    images_txt = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir), key=len):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            num = 0
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    images_txt.append(target + '/' + fname)
                    num += 1
            num_in_class.append(num)

    return images, num_in_class, images_txt



def pil_loader(path, mode='RGB'):

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)

        if mode == 'L':
            return img.convert('L')  # convert image to grey
        elif mode == 'RGB':
            return img.convert('RGB')  # convert image to rgb image
        elif mode == 'HSV':
            return img.convert('HSV')
        # elif mode == 'LAB':
        #     return RGB2Lab(img)


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path, mode):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path, mode)


class MyDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/CGG/xxx.png
        root/CGG/xxy.png
        root/CGG/xxz.png
        root/Real/xxx.png
        root/Real/xxx.png
        root/Real/xxx.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, args, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(args.data_root)

        imgs, num_in_class, images_txt = make_dataset(args.data_root, class_to_idx)

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + args.data_root + "\n"
                                                                                      "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.mode = args.img_mode
        self.input_nc = args.input_nc
        self.imgs = imgs
        self.num_in_class = num_in_class
        self.images_txt = images_txt
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, self.mode)
        if self.transform is not None:
            img = self.transform(img)  # (3,256,256)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.imgs)


class RandomBalancedSampler(Sampler):
    def __init__(self, data_source):
        print('Using RandomBalancedSampler...')
        self.data_source = data_source
        self.num_in_class = data_source.num_in_class

    def __iter__(self):
        num_in_class = self.num_in_class
        a_perm = torch.randperm(num_in_class[0]).tolist()
        b_perm = [x + num_in_class[0] for x in torch.randperm(num_in_class[1]).tolist()]

        if num_in_class[0] > num_in_class[1]:
            a_perm = a_perm[0:num_in_class[1]]
        elif num_in_class[0] < num_in_class[1]:
            b_perm = b_perm[0:num_in_class[0]]

        assert len(a_perm) == len(b_perm)

        index = []
        for tmp in range(len(a_perm)):
            index.append(a_perm[tmp])
            index.append(b_perm[tmp])

        return iter(index)

    def __len__(self):
        return min(self.num_in_class) * 2


# each two element is paired, and order is shuffled for each epoch (shuffle=True)
# the number of samples in two class is same
class PairedSampler(Sampler):
    def __init__(self, data_source):
        print('Using PairedSampler...')
        self.data_source = data_source
        self.num_in_class = data_source.num_in_class

    def __iter__(self):
        num_in_class = self.num_in_class
        a_perm = torch.randperm(num_in_class[0]).tolist()
        b_perm = [x + num_in_class[0] for x in a_perm]

        index = []
        for tmp in range(len(a_perm)):
            index.append(a_perm[tmp])
            index.append(b_perm[tmp])

        return iter(index)

    def __len__(self):
        return min(self.num_in_class) * 2


class DataLoaderHalf(DataLoader):
    def __init__(self, dataset,
                 shuffle=False, batch_size=1, half_constraint=False, sampler_type='RandomBalancedSampler',
                 drop_last=True,
                 num_workers=0, pin_memory=False):
        if half_constraint:
            if sampler_type == 'PairedSampler':
                sampler = PairedSampler(dataset)
            else:
                sampler = RandomBalancedSampler(dataset)
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        super(DataLoaderHalf, self).__init__(dataset, batch_size, None, sampler,
                                             None, num_workers, pin_memory=pin_memory, drop_last=drop_last)

