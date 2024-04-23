""" define data preprocessing/augmentation
"""
from torchvision import transforms

import data.augmentations.basic as aug_basic
import data.augmentations.appearance as aug_appear
import data.augmentations.perspective as aug_persp

import logging
logger = logging.getLogger('root')


def get_transform(args, dataset_name='hypersim', mode='train'):
    assert mode in ['train', 'test']
    logger.info('Defining %s transform for %s dataset' % (mode, dataset_name))
    tf_list = [
        aug_basic.ToTensor(),
    ]

    # RESIZE and CROP
    if mode=='train' and args.data_augmentation_intrins:
        logger.info('Augmentation - Randomize intrinisics (randomly resize & crop the input image)')
        tf_list += [
            aug_basic.RandomIntrins()
        ]
    elif args.input_height == 0 or args.input_width == 0:
        logger.info('Preprocessing - Input images will not be resized')
    elif args.data_augmentation_same_fov == 0:
        logger.info('Preprocessing - Resize to H, W = %s, %s' % (args.input_height, args.input_width))
        tf_list += [
            aug_basic.Resize(H=args.input_height, W=args.input_width),  
        ]
    elif args.data_augmentation_same_fov > 0:
        logger.info('Preprocessing - Set the field-of-view to %s degrees (H, W = %s, %s)' \
                    % (args.data_augmentation_same_fov, args.input_height, args.input_width))
        tf_list += [
            aug_persp.SameFov(new_fov=args.data_augmentation_same_fov, 
                              H=args.input_height, W=args.input_width),
        ]
    if mode == 'train' and args.data_augmentation_crop:
        logger.info('Augmentation - Random crop to H, W = %s, %s' % (args.data_augmentation_crop_height, 
                                                                     args.data_augmentation_crop_width))
        tf_list += [
            aug_basic.RandomCrop(height=args.data_augmentation_crop_height, 
                                 width=args.data_augmentation_crop_width)
        ]
    if mode == 'train' and args.data_augmentation_nyu_crop:
        assert 'nyuv2' in dataset_name
        logger.info('Preprocessing - NYUv2 crop')
        tf_list += [
            aug_basic.NyuCrop()
        ]

    # PERSPECTIVE
    if mode == 'train' and args.data_augmentation_persp:
        logger.info('Augmentation - Perspective augmentation')
        tf_list += [
            aug_persp.RotationAndScale(yaw_range=args.data_augmentation_persp_yaw, 
                                       pitch_range=args.data_augmentation_persp_pitch, 
                                       roll_range=args.data_augmentation_persp_roll, 
                                       random_fov=args.data_augmentation_persp_random_fov, 
                                       min_fov=args.data_augmentation_persp_min_fov, 
                                       max_fov=args.data_augmentation_persp_max_fov,
                                       H=args.input_height, W=args.input_width,
                                       p=1.0),
        ]

    # HORIZONTAL FLIP
    if mode == 'train' and args.data_augmentation_hflip:
        logger.info('Augmentation - Horizontal flip')
        tf_list += [
            aug_basic.HorizontalFlip(p=0.5)
        ]

    # COLOR
    if mode == 'train' and args.data_augmentation_color:
        logger.info('Augmentation - Color augmentation')
        tf_list += [
            aug_basic.ColorAugmentation(gamma_range=(0.9, 1.1), 
                                        brightness_range=(0.75, 1.25), 
                                        color_range=(0.9, 1.1), p=0.5)
        ]

    # APPEARANCE
    if mode == 'train' and args.data_augmentation_appear > 0:
        logger.info('Augmentation - Appearance augmentation (ver %s)' % args.data_augmentation_appear)
        tf_list += [
            aug_appear.AppearAug(p=0.8, ver=args.data_augmentation_appear),
        ]

    tf_list += [
        aug_basic.Normalize(),
        aug_basic.ToDict(),
    ]

    logger.info('Defining %s transform for %s dataset ... DONE' % (mode, dataset_name))
    return transforms.Compose(tf_list)
