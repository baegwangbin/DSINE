""" models from the segmentation_models_pytorch library
    install the library before using these models
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SegModel(nn.Module):
    def __init__(self, architecture='Unet', 
                    encoder_name='mobilenet_v2', 
                    encoder_weights='imagenet',
                    encoder_depth=4,
                    in_channels=3, 
                    classes=4):
        super().__init__()

        """
        architecture = {
            Unet, UnetPlusPlus, EfficientUnetPlusPlus, DeepLabV3, DeepLabV3+
        }
        encoder_name = {
            resnet18 / 11M, resnet34 / 21M, resnet50 / 23M, resnet101 / 42M
            densenet121 / 6M, densenet169 / 12M, densenet201 / 18M
            efficientnet-b0 / 4M, -b1 / 6M, -b2 / 7M, -b3 / 10M, -b4 / 17M, -b5 / 28M
            mobilenet_v2 / 2M
        }
        encoder_weights = {
            "imagenet", None
        }
        in_channels = 3
        classes = 4
        """

        if encoder_depth == 4:
            decoder_channels = (256, 128, 64, 32)

        exec("""self.model = smp.%s(
            encoder_name='%s',
            encoder_weights='%s',
            encoder_depth=%s,
            decoder_channels=%s,
            in_channels=%s,
            classes=%s,
            activation=None,
        )
        """ % (architecture, encoder_name, encoder_weights, encoder_depth, decoder_channels, in_channels, classes))


    def forward(self, x):
        return self.model(x)

