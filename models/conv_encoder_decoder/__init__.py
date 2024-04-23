""" Set of convolutional encoder-decoder architectures
    Activation should be defined in accordance with the task
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger('root')


class ConvEncoderDecoder(nn.Module):
    def __init__(self, architecture, output_dim, **kwargs):
        super(ConvEncoderDecoder, self).__init__()
        logger.info('Model - defining a convolutional encoder decoder ...')
        logger.info('Model - architecture: %s' % architecture)
        logger.info('Model - output_dim: %s' % output_dim)

        if architecture == 'UNet':
            from .unet import UNet
            self.model = UNet(n_classes=output_dim)

        # models from segmentation_models_pytorch library
        # for example, "smp__Unet__resnet50" uses Unet architecture with resnet50 backbone
        elif architecture[:3] == 'smp':
            from .seg_models import SegModel
            _, architecture, encoder_name = architecture.split('__')
            self.model = SegModel(architecture=architecture,
                                  encoder_name=encoder_name,
                                  classes=output_dim)

        # DenseDepth and its variants
        # for example, "densedepth__B5__NF2048__down2__bilinear__BN" means:
        # - "B5":       uses EfficientNet B5 backbone (pretrained)
        # - "NF2048":   bottleneck num_feature = 2048
        # - "down2":    make inference at H/2 X W/2 resolution, then upsample it
        # - "bilinear": uses bilinear upsampling (replace with "learned" to use convex upsampling)
        # - "BN":       uses BatchNorm (replace with "GN" to use GroupNorm)
        elif architecture[:10] == 'densedepth':
            from .dense_depth import DenseDepth
            _, B, NF, down, learned, BN = architecture.split('__')
            self.model = DenseDepth(num_classes=output_dim,
                                    B=int(B[1:2]), pretrained=True,
                                    NF=int(NF[2:]), BN=BN=='BN',
                                    down=int(down.split('down')[1]), learned_upsampling=learned=='learned',
                                    **kwargs)
            
        else:
            raise Exception('architecture not implemented')

        
    def forward(self, img, **kwargs):
        return self.model(img, **kwargs)
