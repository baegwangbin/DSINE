""" appearance augmentations
"""
import cv2
import random
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import Tensor
from typing import List, Tuple


# NOTE: the code below is copied from torchvision
# (https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py)
# See the license at https://github.com/pytorch/vision/blob/main/LICENSE
def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


# NOTE: the code below is copied from torchvision
# (https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py)
# See the license at https://github.com/pytorch/vision/blob/main/LICENSE
def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


# NOTE: the code below is copied from torchvision
# (https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py)
# See the license at https://github.com/pytorch/vision/blob/main/LICENSE
def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d


# NOTE: the code below is copied from torchvision
# (https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py)
# See the license at https://github.com/pytorch/vision/blob/main/LICENSE
def _get_gaussian_kernel2d(kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def line_from_theta(kernel, theta):
    ks = kernel.shape[0]
    if theta < (1/4) * np.pi:
        x = (ks // 2) * np.tan(theta)
        y = (ks // 2)
    elif theta < (3/4) * np.pi:
        x = (ks // 2)
        y = (ks // 2) / np.tan(theta)
    else:
        x = -(ks // 2) * np.tan(theta)
        y = -(ks // 2)

    x1 = int(x + (ks/2.0))
    y1 = int(y + (ks/2.0))

    x2 = int(-x + (ks/2.0))
    y2 = int(-y + (ks/2.0))

    kernel = cv2.line(kernel, (x1, y1), (x2, y2), 1)
    return kernel


class DownUp():
    """ random downsample-and-upsample
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, min_scale=0.5, p=0.1):
        self.min_scale = min_scale
        assert self.min_scale > 0.0 and self.min_scale <= 1.0
        self.p = p
 
    def __call__(self, sample):
        if random.random() < self.p:
            _, orig_H, orig_W = sample.img.shape
            scale = random.uniform(self.min_scale, 1)
            new_H = round(orig_H * scale)
            new_W = round(orig_W * scale)

            sample.img = F.interpolate(sample.img.unsqueeze(0), 
                                       size=(new_H, new_W), 
                                       mode='bilinear', 
                                       align_corners=False,
                                       antialias=True).squeeze(0).clamp(0, 1)

            sample.img = F.interpolate(sample.img.unsqueeze(0), 
                                       size=(orig_H, orig_W), 
                                       mode='bilinear', 
                                       align_corners=False,
                                       antialias=True).squeeze(0).clamp(0, 1)

        return sample


class JpegCompress():
    """ random JPEG compression
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, min_quality=10, max_quality=90, p=0.1):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img_tmp = (sample.img * 255.0).to(dtype=torch.uint8).permute(1, 2, 0).numpy()
            quality = random.randrange(self.min_quality, self.max_quality+1)
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            encoded = cv2.imencode('.jpg', img_tmp, encode_param)[1]
            decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            sample.img = torch.from_numpy(decoded).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
            sample.img = sample.img.clamp(0, 1)
        return sample


class GaussianBlur():
    """ random Gaussian blur
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, ks=11, sigma=(0.1, 10.0), p=0.1):
        self.transform = transforms.GaussianBlur(kernel_size=(ks, ks), sigma=sigma)
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.img = self.transform(sample.img).clamp(0, 1)
        return sample


class MotionBlur():
    """ random motion blur
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, ks=(1, 11), p=0.1):
        self.ks = ks 
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            theta = random.uniform(0, np.pi)
            ks = random.choice(list(range(self.ks[0], self.ks[1]+1, 2)))

            # line
            kernel = np.zeros((ks, ks), np.float32)
            kernel = line_from_theta(kernel, theta)
            kernel = torch.from_numpy(kernel)

            # (ks, ks)
            kernel /= torch.sum(kernel)
            kernel = kernel.expand(sample.img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

            sample.img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(sample.img, [kernel.dtype])
            padding = [ks // 2, ks // 2, ks // 2, ks // 2]
            sample.img = F.pad(sample.img, padding, mode="reflect")
            sample.img = F.conv2d(sample.img, kernel, groups=sample.img.shape[-3])
            sample.img = _cast_squeeze_out(sample.img, need_cast, need_squeeze, out_dtype)
            sample.img = sample.img.clamp(0, 1)
        return sample


class GaussianNoise():
    """ random Gaussian noise
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, sigma=(0.01, 0.05), p=0.1):
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            sample.img = sample.img + sigma * torch.randn(sample.img.size())
            sample.img = sample.img.clamp(0, 1)
        return sample


class ColorJitter():
    """ random Color Jitter
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=0.5):
        self.transform = transforms.ColorJitter(brightness=brightness, 
                                                contrast=contrast, 
                                                saturation=saturation, 
                                                hue=hue)
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.img = self.transform(sample.img).clamp(0, 1)
        return sample


class Grayscale():
    """ random Grayscale
        sample.img is a torch tensor of shape (3, H, W), normalized to [0, 1]
    """
    def __init__(self, p=0.1):
        self.transform = transforms.Grayscale(num_output_channels=3)
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.img = self.transform(sample.img).clamp(0, 1)
        return sample


class AppearAug():
    def __init__(self, p=0.1, ver=0):
        self.p = p    

        if ver == 1:
            # mild
            self.transforms = transforms.Compose([
                DownUp(min_scale=0.2, p=0.1),
                JpegCompress(min_quality=10, max_quality=90, p=0.1),
                GaussianBlur(ks=7, sigma=(0.1, 10.0), p=0.1),
                GaussianNoise(sigma=(0.02, 0.02), p=0.1),
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.1),
                Grayscale(p=0.01),            
            ])

        elif ver == 2:
            # mild + motion blur
            self.transforms = transforms.Compose([
                DownUp(min_scale=0.2, p=0.1),
                JpegCompress(min_quality=10, max_quality=90, p=0.1),
                GaussianBlur(ks=11, sigma=(0.1, 5.0), p=0.1),
                MotionBlur(ks=(1,11), p=0.1),
                GaussianNoise(sigma=(0.01, 0.05), p=0.1),
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.1),
                Grayscale(p=0.01),  
            ])

    def __call__(self, sample):
        if random.random() < self.p:
            sample = self.transforms(sample)
        return sample
