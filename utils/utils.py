import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

import logging
logger = logging.getLogger('root')


def load_checkpoint(fpath, model):
    assert os.path.exists(fpath)
    logger.info('loading checkpoint... %s' % fpath)
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    logger.info('loading checkpoint... / done')
    return model


def save_model(model, target_path, total_iter):
    torch.save({"model": model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'lr_scheduler_state_dict': scheduler.state_dict(),
                "iter": total_iter
                }, target_path)
    logger.info('model saved / path: {}'.format(target_path))


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def save_args(args, filename):
    with open(filename, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def txt_to_list(txt_path):
    with open(txt_path, 'r') as f:
        content = f.readlines()
        content = [i.strip() for i in content]
    return content


def setup_custom_logger(name, test=False):
    formatter = logging.Formatter(fmt='[%(asctime)s]- %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger(name)
    if test:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def change_logger_dest(logger, new_dest):
    formatter = logging.Formatter(fmt='[%(asctime)s]- %(levelname)s - %(module)s - %(message)s')
    handler = logging.FileHandler(new_dest, mode='a')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value, count_add=1):
        self.avg = (count_add * value + self.count * self.avg) / (count_add + self.count)
        self.count += count_add

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict, count_add):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value, count_add)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_normal_error(pred_norm, gt_norm):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    return pred_error


def compute_normal_metrics(total_normal_errors):
    """ compute surface normal metrics (used for benchmarking)
        NOTE: total_normal_errors should be a 1D torch tensor of errors in degrees
    """
    total_normal_errors = total_normal_errors.detach().cpu().numpy()
    num_pixels = total_normal_errors.shape[0]

    metrics = {
        'mean': np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse': np.sqrt(np.sum(total_normal_errors * total_normal_errors) / num_pixels),
        'a1': 100.0 * (np.sum(total_normal_errors < 5) / num_pixels),
        'a2': 100.0 * (np.sum(total_normal_errors < 7.5) / num_pixels),
        'a3': 100.0 * (np.sum(total_normal_errors < 11.25) / num_pixels),
        'a4': 100.0 * (np.sum(total_normal_errors < 22.5) / num_pixels),
        'a5': 100.0 * (np.sum(total_normal_errors < 30) / num_pixels)
    }
    return metrics


def compute_normal_metrics2(total_normal_errors):
    """ compute surface normal metrics (used for validation)
        NOTE: total_normal_errors should be a 1D torch tensor of errors in degrees
    """
    num_pixels = total_normal_errors.shape[0]

    metrics = {
        'mean': torch.mean(total_normal_errors).item(),
        'mse': torch.mean(total_normal_errors * total_normal_errors).item(),
        'a1': 100.0 * torch.mean((total_normal_errors < 5).float()).item(),
        'a2': 100.0 * torch.mean((total_normal_errors < 7.5).float()).item(),
        'a3': 100.0 * torch.mean((total_normal_errors < 11.25).float()).item(),
        'a4': 100.0 * torch.mean((total_normal_errors < 22.5).float()).item(),
        'a5': 100.0 * torch.mean((total_normal_errors < 30).float()).item(),
    }
    return metrics, num_pixels



def get_padding(orig_H, orig_W):
    """ returns how the input of shape (orig_H, orig_W) should be padded
        this ensures that both H and W are divisible by 32
    """
    if orig_W % 32 == 0:
        l = 0
        r = 0
    else:
        new_W = 32 * ((orig_W // 32) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 32 == 0:
        t = 0
        b = 0
    else:
        new_H = 32 * ((orig_H // 32) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b


def pad_input(img, intrins, lrtb=(0,0,0,0)):
    """ pad input image
        img should be a torch tensor of shape (B, 3, H, W)
        intrins should be a torch tensor of shape (B, 3, 3)
    """
    l, r, t, b = lrtb
    if l+r+t+b != 0:
        pad_value_R = (0 - 0.485) / 0.229
        pad_value_G = (0 - 0.456) / 0.224
        pad_value_B = (0 - 0.406) / 0.225

        img_R = F.pad(img[:,0:1,:,:], (l, r, t, b), mode="constant", value=pad_value_R)
        img_G = F.pad(img[:,1:2,:,:], (l, r, t, b), mode="constant", value=pad_value_G)
        img_B = F.pad(img[:,2:3,:,:], (l, r, t, b), mode="constant", value=pad_value_B)

        img = torch.cat([img_R, img_G, img_B], dim=1)

        if intrins is not None:
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t
    return img, intrins


