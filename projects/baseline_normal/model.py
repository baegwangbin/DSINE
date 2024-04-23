import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger('root')


def normal_activation(out):
    normal, kappa = out[:, :3, :, :], out[:, 3:, :, :]
    normal = F.normalize(normal, p=2, dim=1)
    kappa = F.elu(kappa) + 1.0
    return torch.cat([normal, kappa], dim=1)


# N-Net
class NNET(nn.Module):
    def __init__(self, args):
        super(NNET, self).__init__()
        arch = args.NNET_architecture
        logger.info('Model architecture: %s' % arch)

        from models.conv_encoder_decoder import ConvEncoderDecoder
        self.n_net = ConvEncoderDecoder(
            architecture=args.NNET_architecture,
            output_dim=args.NNET_output_dim,
            activation_fn=normal_activation
        )

        if arch[:10]=='densedepth':
            logger.info('activation will be done within the architecture')
            self.activation_fn = lambda a: a
        else:
            self.activation_fn = normal_activation

    def forward(self, img, **kwargs):
        return self.activation_fn(self.n_net(img, **kwargs))



