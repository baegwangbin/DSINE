import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger('root')

COS_EPS = 1e-7


# define loss function
def define_loss(loss_name):
    if loss_name == 'L1':
        return l1_loss
    elif loss_name == 'L2':
        return l2_loss
    elif loss_name == 'AL':
        return angular_loss
    elif loss_name == 'NLL_vonmf':
        return vonmf_loss
    elif loss_name == 'NLL_angmf':
        return angmf_loss
    else:
        raise Exception('invalid loss fn name: %s' % loss_name)


# loss_name = "L1"
def l1_loss(norm_out, gt_norm, gt_norm_mask): 
    """ norm_out:       (B, 3, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)
    """
    pred_norm = norm_out[:, 0:3, ...]

    l1 = torch.sum(torch.abs(gt_norm - pred_norm), dim=1, keepdim=True)      # (B, 1, ...)
    l1 = l1[gt_norm_mask]
    return torch.mean(l1)


# loss_name = "L2"
def l2_loss(norm_out, gt_norm, gt_norm_mask): 
    """ norm_out:       (B, 3, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)   
    """
    pred_norm = norm_out[:, 0:3, ...]

    l2 = torch.sum(torch.square(gt_norm - pred_norm), dim=1, keepdim=True)   # (B, 1, ...)
    l2 = l2[gt_norm_mask]
    return torch.mean(l2)


# loss_name = "AL"
def angular_loss(norm_out, gt_norm, gt_norm_mask): 
    """ norm_out:       (B, 3, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)   

    """
    pred_norm = norm_out[:, 0:3, ...]
    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)    
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)
    angle = torch.acos(dot[valid_mask])
    return torch.mean(angle)


# loss_name = "NLL_vonmf"
def nll_vonmf(dot, pred_kappa):
    loss = - torch.log(pred_kappa) \
            - (pred_kappa * (dot - 1)) \
            + torch.log(1 - torch.exp(- 2 * pred_kappa))
    return loss

def vonmf_loss(norm_out, gt_norm, gt_norm_mask):
    """ norm_out:       (B, 4, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)   
    """
    pred_norm, pred_kappa = norm_out[:, 0:3, ...], norm_out[:, 3:, ...]

    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)    
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)

    # compute the loss
    nll = nll_vonmf(dot[valid_mask], pred_kappa[valid_mask])
    return torch.mean(nll)


# loss_name = "NLL_angmf"
def nll_angmf(dot, pred_kappa):
    loss = - torch.log(torch.square(pred_kappa) + 1) \
            + pred_kappa * torch.acos(dot) \
            + torch.log(1 + torch.exp(-pred_kappa * np.pi))
    return loss

def angmf_loss(norm_out, gt_norm, gt_norm_mask):
    """ norm_out:       (B, 4, ...)
        gt_norm:        (B, 3, ...)
        gt_norm_mask:   (B, 1, ...)   
    """
    pred_norm, pred_kappa = norm_out[:, 0:3, ...], norm_out[:, 3:, ...]

    dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1).unsqueeze(1)
    valid_mask = torch.logical_and(gt_norm_mask, torch.abs(dot.detach()) < 1-COS_EPS)

    # compute the loss
    nll = nll_angmf(dot[valid_mask], pred_kappa[valid_mask])
    return torch.mean(nll)


# compute loss for baseline_normal experiments
class ComputeLoss(nn.Module):
    def __init__(self, args):
        """ args.loss_fn can be one of following:
            - L1            - L1 loss       (no uncertainty)
            - L2            - L2 loss       (no uncertainty)
            - AL            - Angular loss  (no uncertainty)
            - NLL_vonmf     - NLL of vonMF distribution
            - NLL_angmf     - NLL of Angular vonMF distribution (from "Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation", ICCV 2021)
        """
        super(ComputeLoss, self).__init__()
        logger.info('Loss: %s' % args.loss_fn)
       
        # define pixel-wise loss fn
        self.loss_name = loss_name = args.loss_fn
        self.loss_fn = define_loss(loss_name)

    def forward(self, norm_out, gt_norm, gt_norm_mask):
        """ norm_out:       (B, 3, ...) or (B, 4, ...)   ...   pred_norm should already be L2 normalized / pred_kappa should be positive
            gt_norm:        (B, 3, ...)
            gt_norm_mask:   (B, 1, ...)
        """
        loss = self.loss_fn(norm_out, gt_norm, gt_norm_mask)
        return loss
