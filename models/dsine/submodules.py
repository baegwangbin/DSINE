""" submodules for DSINE
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pixel_coords(h, w):
    # pixel array (1, 2, H, W)
    pixel_coords = np.ones((3, h, w)).astype(np.float32)
    x_range = np.concatenate([np.arange(w).reshape(1, w)] * h, axis=0)
    y_range = np.concatenate([np.arange(h).reshape(h, 1)] * w, axis=1)
    pixel_coords[0, :, :] = x_range + 0.5
    pixel_coords[1, :, :] = y_range + 0.5
    return torch.from_numpy(pixel_coords).unsqueeze(0)


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, ks=3):
        super(ConvGRU, self).__init__()
        p = (ks - 1) // 2
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h
    

def get_unfold(pred_norm, ps, pad):
    B, C, H, W = pred_norm.shape
    pred_norm = F.pad(pred_norm, pad=(pad,pad,pad,pad), mode='replicate')       # (B, C, h, w)
    pred_norm_unfold = F.unfold(pred_norm, [ps, ps], padding=0)                 # (B, C X ps*ps, h*w)
    pred_norm_unfold = pred_norm_unfold.view(B, C, ps*ps, H, W)                 # (B, C, ps*ps, h, w)
    return pred_norm_unfold


def normal_activation(out, elu_kappa=True):
    normal, kappa = out[:, :3, :, :], out[:, 3:, :, :]
    normal = F.normalize(normal, p=2, dim=1)
    if elu_kappa:
        kappa = F.elu(kappa) + 1.0
    return torch.cat([normal, kappa], dim=1)


class RayReLU(nn.Module):
    def __init__(self, eps=1e-2):
        super(RayReLU, self).__init__()
        self.eps = eps

    def forward(self, pred_norm, ray):
        # angle between the predicted normal and ray direction
        cos = torch.cosine_similarity(pred_norm, ray, dim=1).unsqueeze(1) # (B, 1, H, W)

        # component of pred_norm along view
        norm_along_view = ray * cos

        # cos should be bigger than eps
        norm_along_view_relu = ray * (torch.relu(cos - self.eps) + self.eps)

        # difference        
        diff = norm_along_view_relu - norm_along_view

        # updated pred_norm
        new_pred_norm = pred_norm + diff
        new_pred_norm = F.normalize(new_pred_norm, dim=1)

        return new_pred_norm  