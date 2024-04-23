""" basic augmentations
"""
import random
import numpy as np

import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import logging
logger = logging.getLogger('root')


def resize(sample, new_H, new_W):
    _, orig_H, orig_W = sample.img.shape
    sample.img = F.interpolate(sample.img.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False, antialias=True).squeeze(0)
    if sample.depth is not None:
        sample.depth = F.interpolate(sample.depth.unsqueeze(0), size=(new_H, new_W), mode='nearest').squeeze(0)
    if sample.depth_mask is not None:
        sample.depth_mask = F.interpolate(sample.depth_mask.unsqueeze(0).float(), size=(new_H, new_W), mode='nearest').squeeze(0) > 0.5
    if sample.normal is not None:
        sample.normal = F.interpolate(sample.normal.unsqueeze(0), size=(new_H, new_W), mode='nearest').squeeze(0)
    if sample.normal_mask is not None:
        sample.normal_mask = F.interpolate(sample.normal_mask.unsqueeze(0).float(), size=(new_H, new_W), mode='nearest').squeeze(0) > 0.5
    if sample.intrins is not None:
        # NOTE: top-left is (0,0)
        sample.intrins[0, 0] = sample.intrins[0, 0] * (new_W / orig_W)     # fx
        sample.intrins[1, 1] = sample.intrins[1, 1] * (new_H / orig_H)     # fy
        sample.intrins[0, 2] = (sample.intrins[0, 2] + 0.5) * (new_W / orig_W) - 0.5     # cx
        sample.intrins[1, 2] = (sample.intrins[1, 2] + 0.5) * (new_H / orig_H) - 0.5     # cy
    return sample


def pad(sample, lrtb):
    l, r, t, b = lrtb
    sample.img = F.pad(sample.img, (l, r, t, b), mode="constant", value=0)
    if sample.depth is not None:
        sample.depth = F.pad(sample.depth, (l, r, t, b), mode="constant", value=0)
    if sample.depth_mask is not None:
        sample.depth_mask = F.pad(sample.depth_mask, (l, r, t, b), mode="constant", value=False)
    if sample.normal is not None:
        sample.normal = F.pad(sample.normal, (l, r, t, b), mode="constant", value=0)
    if sample.normal_mask is not None:
        sample.normal_mask = F.pad(sample.normal_mask, (l, r, t, b), mode="constant", value=False)
    if sample.intrins is not None:
        sample.intrins[0, 2] = sample.intrins[0, 2] + l
        sample.intrins[1, 2] = sample.intrins[1, 2] + t
    return sample


def crop(sample, y, H, x, W):
    sample.img = sample.img[:, y:y+H, x:x+W]
    if sample.depth is not None:
        sample.depth = sample.depth[:, y:y+H, x:x+W]
    if sample.depth_mask is not None:
        sample.depth_mask = sample.depth_mask[:, y:y+H, x:x+W]
    if sample.normal is not None:
        sample.normal = sample.normal[:, y:y+H, x:x+W]
    if sample.normal_mask is not None:
        sample.normal_mask = sample.normal_mask[:, y:y+H, x:x+W]
    if sample.intrins is not None:
        sample.intrins[0, 2] = sample.intrins[0, 2] - x
        sample.intrins[1, 2] = sample.intrins[1, 2] - y
    return sample


class ToTensor():
    """ numpy arrays to torch tensors
    """
    def __call__(self, sample):
        sample.img = torch.from_numpy(sample.img).permute(2, 0, 1)                      # (3, H, W)
        if sample.depth is not None:
            sample.depth = torch.from_numpy(sample.depth).permute(2, 0, 1)              # (1, H, W)
        if sample.depth_mask is not None:
            sample.depth_mask = torch.from_numpy(sample.depth_mask).permute(2, 0, 1)    # (1, H, W)
        if sample.normal is not None:
            sample.normal = torch.from_numpy(sample.normal).permute(2, 0, 1)            # (3, H, W)
        if sample.normal_mask is not None:
            sample.normal_mask = torch.from_numpy(sample.normal_mask).permute(2, 0, 1)  # (1, H, W)
        if sample.intrins is not None:
            sample.intrins = torch.from_numpy(sample.intrins)                           # (3, 3)
        return sample


class RandomIntrins():
    """ randomize intrinsics
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __call__(self, sample):
        assert 'crop_H' in sample.info.keys() 
        assert 'crop_W' in sample.info.keys()
        crop_H = sample.info['crop_H']
        crop_W = sample.info['crop_W']

        # height-based resizing
        _, orig_H, orig_W = sample.img.shape
        new_H = random.randrange(min(orig_H, crop_H), max(orig_H, crop_H)+1)
        new_W = round((new_H / orig_H) * orig_W)
        sample = resize(sample, new_H=new_H, new_W=new_W)

        # pad if necessary
        orig_H, orig_W = sample.img.shape[1], sample.img.shape[2]
        l, r, t, b = 0, 0, 0, 0
        if crop_H > orig_H:
            t = b = crop_H - orig_H
        if crop_W > orig_W:
            l = r = crop_W - orig_W
        sample = pad(sample, (l, r, t, b))

        # crop
        assert sample.img.shape[1] >= crop_H
        assert sample.img.shape[2] >= crop_W
        x = random.randint(0, sample.img.shape[2] - crop_W)
        y = random.randint(0, sample.img.shape[1] - crop_H)
        sample = crop(sample, y=y, H=crop_H, x=x, W=crop_W)

        return sample


class Resize():
    """ resize to (H, W)
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, H=480, W=640):
        self.H = H
        self.W = W

    def __call__(self, sample):
        return resize(sample, new_H=self.H, new_W=self.W)


class RandomCrop():
    """ random crop
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, H=416, W=544):
        self.H = H
        self.W = W

    def __call__(self, sample):
        assert sample.img.shape[1] >= self.H
        assert sample.img.shape[2] >= self.W
        x = random.randint(0, sample.img.shape[2] - self.W)
        y = random.randint(0, sample.img.shape[1] - self.H)
        return crop(sample, y=y, H=self.H, x=x, W=self.W)
    

class NyuCrop():
    """ crop image border for NYUv2 images
        W = 43:608 / H = 45:472
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __call__(self, sample):
        return crop(sample, y=45, H=472-45, x=43, W=608-43)


class HorizontalFlip():
    """ random horizontal flipping
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.img = TF.hflip(sample.img)
            if sample.depth is not None:
                sample.depth = TF.hflip(sample.depth)
            if sample.depth_mask is not None:
                sample.depth_mask = TF.hflip(sample.depth_mask)
            if sample.normal is not None:
                sample.normal = TF.hflip(sample.normal)
                sample.normal[0, :, :] = -sample.normal[0, :, :]
            if sample.normal_mask is not None:
                sample.normal_mask = TF.hflip(sample.normal_mask)
            if sample.intrins is not None:
                # NOTE: top-left is (0,0)
                _, H, W = sample.img.shape
                sample.intrins[0, 2] = sample.intrins[0, 2] + 0.5   # top-left is (0.5, 0.5)
                sample.intrins[0, 2] = W - sample.intrins[0, 2]
                sample.intrins[0, 2] = sample.intrins[0, 2] - 0.5   # top-left is (0, 0)
            sample.flipped = True
        return sample


class ColorAugmentation():
    """ color augmentation
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, gamma_range=(0.9, 1.1),
                       brightness_range=(0.75, 1.25),
                       color_range=(0.9, 1.1),
                       p=0.5):
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.color_range = color_range
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            # gamma augmentation
            gamma = random.uniform(*self.gamma_range)
            sample.img = sample.img ** gamma

            # brightness augmentation
            brightness = random.uniform(*self.brightness_range)
            sample.img = sample.img * brightness

            # color augmentation
            colors = np.random.uniform(*self.color_range, size=3).astype(np.float32)
            colors = torch.from_numpy(colors).view(3, 1, 1)
            sample.img = sample.img * colors

            # clip
            sample.img = torch.clip(sample.img, 0, 1)

        return sample
    

class Normalize():
    """ mean & std: for image normalization
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        sample.img = self.normalize(torch.clip(sample.img, min=0.0, max=1.0))
        return sample


class ToDict():
    def __call__(self, sample):
        data_dict = {}
        for k, v in vars(sample).items():
            if v is not None:
                data_dict[k] = v
        return data_dict
