import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger('root')

from projects.baseline_normal.losses import define_loss


# compute loss for DSINE experiments
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
        logger.info('Loss: %s / gamma: %s' % (args.loss_fn, args.loss_gamma))

        # define pixel-wise loss fn
        self.loss_name = loss_name = args.loss_fn
        self.loss_fn = define_loss(loss_name)
        self.gamma = args.loss_gamma

    def forward(self, pred_list, gt_norm, gt_norm_mask):
        n_predictions = len(pred_list)
        loss = 0.0
        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            norm_out = pred_list[i]
            loss = loss + i_weight * self.loss_fn(norm_out, gt_norm, gt_norm_mask)
        return loss
