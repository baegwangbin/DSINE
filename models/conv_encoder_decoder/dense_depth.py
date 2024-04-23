""" The code below is an adaptation of DenseDepth (https://github.com/ialhashim/DenseDepth)
    We replaced the BatchNorm with GroupNorm, 
    added weight standardization, 
    added convex upsampling layer, 
    and allowed the backbone and decoder depth to be changeable
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_encoder_decoder.submodules import Encoder, Decoder


class DenseDepth(nn.Module):
    def __init__(self, num_classes,
                 B=5, pretrained=True,
                 NF=2048, BN=True,
                 down=2, learned_upsampling=True,
                 **kwargs):
        super(DenseDepth, self).__init__()
        self.encoder = Encoder(B=B, pretrained=pretrained)
        self.decoder = Decoder(num_classes=num_classes,
                               B=B, NF=NF, BN=BN,
                               down=down, learned_upsampling=learned_upsampling,
                               **kwargs)

    def forward(self, x, **kwargs):
        return self.decoder(self.encoder(x), **kwargs)

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder]
        for m in modules:
            yield from m.parameters()

