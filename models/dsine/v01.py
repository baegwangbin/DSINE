""" DSINE_v01
    - (O) ray direction encoding
    - (X) rotation estimation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_encoder_decoder.submodules import Encoder, UpSampleBN, UpSampleGN, \
    INPUT_CHANNELS_DICT, upsample_via_bilinear, upsample_via_mask, get_prediction_head
from models.dsine.submodules import normal_activation, get_pixel_coords

import logging
logger = logging.getLogger('root')


class DSINE_v01(nn.Module):
    def __init__(self, args):
        super(DSINE_v01, self).__init__()
        B = args.NNET_encoder_B
        NF = args.NNET_decoder_NF
        BN = args.NNET_decoder_BN
        down = args.NNET_decoder_down
        learned_upsampling = args.NNET_learned_upsampling

        logger.info('Defining DSINE_v01 (baseline encoder-decoder, w/ ray encoding)')
        logger.info('B: %s / NF: %s / BN: %s'  % (B, NF, BN))
        logger.info('output_dim: %s / down: %s / learned upsampling: %s' % (args.NNET_output_dim, down, learned_upsampling))
        self.encoder = Encoder(B=B, pretrained=True)
        self.decoder = Decoder(num_classes=args.NNET_output_dim,
                               B=B, NF=NF, BN=BN,
                               down=down, learned_upsampling=learned_upsampling)

    def forward(self, x, **kwargs):
        return self.decoder(self.encoder(x), **kwargs)

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder]
        for m in modules:
            yield from m.parameters()


class Decoder(nn.Module):
    def __init__(self, num_classes=3, B=5, NF=2048, BN=False,
                 down=8, learned_upsampling=True):
        super(Decoder, self).__init__()
        input_channels = INPUT_CHANNELS_DICT[B]

        # use BN or GN
        UpSample = UpSampleBN if BN else UpSampleGN

        features = NF
        self.conv2 = nn.Conv2d(input_channels[0] + 2, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSample(skip_input=features // 1 + input_channels[1] + 2, output_features=features // 2, align_corners=False)
        self.up2 = UpSample(skip_input=features // 2 + input_channels[2] + 2, output_features=features // 4, align_corners=False)

        if down == 8:
            i_dim = features // 4
        elif down == 4:
            self.up3 = UpSample(skip_input=features // 4 + input_channels[3] + 2, output_features=features // 8, align_corners=False)
            i_dim = features // 8
        elif down == 2:
            self.up3 = UpSample(skip_input=features // 4 + input_channels[3] + 2, output_features=features // 8, align_corners=False)
            self.up4 = UpSample(skip_input=features // 8 + input_channels[4] + 2, output_features=features // 16, align_corners=False)
            i_dim = features // 16
        else:
            raise Exception('invalid downsampling ratio')

        self.downsample_ratio = down
        self.output_dim = num_classes

        self.pred_head = get_prediction_head(i_dim+2, 128, num_classes)
        if learned_upsampling:
            self.mask_head = get_prediction_head(i_dim+2, 128, 9 * self.downsample_ratio * self.downsample_ratio)
            self.upsample_fn = upsample_via_mask
        else:
            self.mask_head = lambda a: None
            self.upsample_fn = upsample_via_bilinear

        # pixel coordinates (1, 2, H, W)
        self.pixel_coords = get_pixel_coords(h=2000, w=2000).to(0)

    def ray_embedding(self, x, intrins, orig_H, orig_W):
        B, _, H, W = x.shape
        fu = intrins[:, 0, 0].unsqueeze(-1).unsqueeze(-1) * (W / orig_W)
        cu = intrins[:, 0, 2].unsqueeze(-1).unsqueeze(-1) * (W / orig_W)
        fv = intrins[:, 1, 1].unsqueeze(-1).unsqueeze(-1) * (H / orig_H)
        cv = intrins[:, 1, 2].unsqueeze(-1).unsqueeze(-1) * (H / orig_H)

        # (B, 2, H, W)
        uv = self.pixel_coords[:, :2, :H, :W].repeat(B, 1, 1, 1)
        uv[:, 0, :, :] = (uv[:, 0, :, :] - cu) / fu
        uv[:, 1, :, :] = (uv[:, 1, :, :] - cv) / fv
        return torch.cat([x, uv], dim=1)

    def forward(self, features, intrins, mode='train'):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]
        _, _, orig_H, orig_W = features[0].shape
        
        # STEP 1: make top-left pixel (0.5, 0.5)
        intrins[:, 0, 2] += 0.5
        intrins[:, 1, 2] += 0.5

        x_d0 = self.conv2(self.ray_embedding(x_block4, intrins, orig_H, orig_W))
        x_d1 = self.up1(x_d0, self.ray_embedding(x_block3, intrins, orig_H, orig_W))

        if self.downsample_ratio == 8:
            x_feat = self.up2(x_d1, self.ray_embedding(x_block2, intrins, orig_H, orig_W))
        elif self.downsample_ratio == 4:
            x_d2 = self.up2(x_d1, self.ray_embedding(x_block2, intrins, orig_H, orig_W))
            x_feat = self.up3(x_d2, self.ray_embedding(x_block1, intrins, orig_H, orig_W))
        elif self.downsample_ratio == 2:
            x_d2 = self.up2(x_d1, self.ray_embedding(x_block2, intrins, orig_H, orig_W))
            x_d3 = self.up3(x_d2, self.ray_embedding(x_block1, intrins, orig_H, orig_W))
            x_feat = self.up4(x_d3, self.ray_embedding(x_block0, intrins, orig_H, orig_W))

        out = self.pred_head(self.ray_embedding(x_feat, intrins, orig_H, orig_W))
        out = normal_activation(out, elu_kappa=True)

        mask = self.mask_head(self.ray_embedding(x_feat, intrins, orig_H, orig_W))
        up_out = self.upsample_fn(out, mask, self.downsample_ratio)
        up_out = normal_activation(up_out, elu_kappa=False)
        return [up_out]

