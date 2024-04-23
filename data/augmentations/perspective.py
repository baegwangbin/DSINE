""" perspective augmentations
"""
import random
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from utils.rotation import get_R
from utils.projection import get_ray_array, intrins_from_fov, pix_to_src_coords, zbuffer_to_radial, radial_to_zbuffer


class SameFov():
    """ Set the Field-of-View to be the same for all images
    """
    def __init__(self, new_fov=60, H=480, W=640):
        self.new_intrins = intrins_from_fov(new_fov, H, W)
        self.new_H = H
        self.new_W = W

    def __call__(self, sample):
        _, H, W = sample.img.shape

        # (3, H*W)
        tgt_ray = get_ray_array(H=self.new_H, W=self.new_W, intrins=self.new_intrins.unsqueeze(0)).squeeze(0)   # (3, H*W)
        src_pix = sample.intrins.matmul(tgt_ray)                                                                # (3, H*W)
        src_coords = pix_to_src_coords(src_pix, 
                                       new_H=self.new_H, new_W=self.new_W, 
                                       orig_H=H, orig_W=W)

        # image (1, 3, H, W) / normal (1, 3, H, W) / normal_mask (1, 1, H, W)
        img_warped = F.grid_sample(sample.img.unsqueeze(0), src_coords, mode='bilinear', padding_mode='zeros', align_corners=False)
        sample.img = img_warped.squeeze(0)
        if sample.depth is not None:
            depth_warped = F.grid_sample(sample.depth.unsqueeze(0), src_coords, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0)            
            sample.depth = depth_warped
        if sample.depth_mask is not None:
            depth_mask_warped = F.grid_sample(sample.depth_mask.float().unsqueeze(0), src_coords, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0) > 0.5
            sample.depth_mask = depth_mask_warped
        if sample.normal is not None:
            normal_warped = F.grid_sample(sample.normal.unsqueeze(0), src_coords, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0)
            sample.normal = normal_warped
        if sample.normal_mask is not None:
            normal_mask_warped = F.grid_sample(sample.normal_mask.float().unsqueeze(0), src_coords, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0) > 0.5
            sample.normal_mask = normal_mask_warped
        
        sample.intrins = self.new_intrins.clone()
        return sample


class RotationAndScale():
    """ Perspective augmentation
    """
    def __init__(self, 
                 yaw_range=30, pitch_range=30, roll_range=30, 
                 random_fov=True, min_fov=60, max_fov=90,
                 H=None, W=None,
                 p=0.5):
        self.no_rotation = False
        if yaw_range == 0 and pitch_range == 0 and roll_range == 0:
            print('NO ROTATION')
            self.no_rotation = True

        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.roll_range = roll_range

        self.random_fov = random_fov
        self.min_fov = min_fov
        self.max_fov = max_fov

        self.H = H
        self.W = W

        self.p = p

    def sample_rotation(self):
        # x, y, z = pitch, yaw, roll
        if self.no_rotation:
            R = torch.eye(3, dtype=torch.float32)
            R_inv = torch.eye(3, dtype=torch.float32)
        else:
            yaw = np.radians(random.uniform(-self.yaw_range, self.yaw_range))
            pitch = np.radians(random.uniform(-self.pitch_range, self.pitch_range))
            roll = np.radians(random.uniform(-self.roll_range, self.roll_range))

            R, R_inv = get_R(yaw, pitch, roll)
            R = torch.from_numpy(R.astype(np.float32))
            R_inv= torch.from_numpy(R_inv.astype(np.float32))
        return R, R_inv

    def __call__(self, sample):
        if random.random() < self.p:
            if self.H is None:
                _, new_H, new_W = sample.img.shape
            else:
                new_H, new_W = self.H, self.W
            _, orig_H, orig_W = sample.img.shape
            assert not sample.flipped

            # sample rotation matrix
            R, R_inv = self.sample_rotation()

            # randomize fov
            if self.random_fov:
                new_fov = random.uniform(self.min_fov, self.max_fov)
                new_intrins = intrins_from_fov(new_fov, new_H, new_W)
            else:
                new_intrins = sample.intrins

            # (3, H*W)
            tgt_ray = get_ray_array(new_H, new_W, new_intrins.unsqueeze(0)).squeeze(0)
            src_pix = sample.intrins.matmul(R_inv).matmul(tgt_ray)

            # (1, H, W, 2)
            src_coords = pix_to_src_coords(src_pix, new_H=new_H, new_W=new_W, orig_H=orig_H, orig_W=orig_W)

            # image (1, 3, H, W) / normal (1, 3, H, W) / normal_mask (1, 1, H, W)
            img_warped = F.grid_sample(sample.img.unsqueeze(0), src_coords, mode='bilinear', padding_mode='zeros', align_corners=False)
            sample.img = img_warped.squeeze(0)

            if sample.depth is not None:
                radial = zbuffer_to_radial(sample.depth.unsqueeze(0), sample.intrins.unsqueeze(0), orig_H, orig_W)
                radial_warped = F.grid_sample(radial, src_coords, mode='nearest', padding_mode='zeros', align_corners=False)
                depth_warped = radial_to_zbuffer(radial_warped, new_intrins.unsqueeze(0), new_H, new_W).squeeze(0)
                sample.depth = depth_warped
            
            if sample.depth_mask is not None:
                depth_mask_warped = F.grid_sample(sample.depth_mask.float().unsqueeze(0), src_coords, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0) > 0.5
                sample.depth_mask = depth_mask_warped

            if sample.normal is not None:
                normal_warped = F.grid_sample(sample.normal.unsqueeze(0), src_coords, mode='nearest', padding_mode='zeros', align_corners=False)
                normal_warped = R.matmul(normal_warped.squeeze(0).reshape(3, -1)).reshape(3, new_H, new_W)
                sample.normal = normal_warped
            
            if sample.normal_mask is not None:
                normal_mask_warped = F.grid_sample(sample.normal_mask.float().unsqueeze(0), src_coords, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0) > 0.5
                sample.normal_mask = normal_mask_warped
            
            sample.intrins = new_intrins

        return sample

