""" Get samples from ScanNet (https://github.com/ScanNet/ScanNet)
    NOTE: GT surface normals and data split are from FrameNet (ICCV 2019) - https://github.com/hjwdzh/FrameNet
"""
import os
import cv2
import numpy as np

from data import Sample

from projects import DATASET_DIR
DATASET_PATH = os.path.join(DATASET_DIR, 'dsine_eval', 'scannet')


def get_sample(args, sample_path, info):
    # e.g. sample_path = "scene0532_00/000000_img.png"
    scene_name = sample_path.split('/')[0]
    img_name, img_ext = sample_path.split('/')[1].split('_img')

    img_path = '%s/%s' % (DATASET_PATH, sample_path)
    normal_path = img_path.replace('_img'+img_ext, '_normal.png')
    intrins_path = img_path.replace('_img'+img_ext, '_intrins.npy')
    assert os.path.exists(img_path)
    assert os.path.exists(normal_path)
    assert os.path.exists(intrins_path)

    # read image (H, W, 3)
    img = None
    if args.load_img:
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

    # read normal (H, W, 3)
    normal = normal_mask = None
    if args.load_normal:
        normal = cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        normal_mask = np.sum(normal, axis=2, keepdims=True) > 0
        normal = (normal.astype(np.float32) / 255.0) * 2.0 - 1.0

    # read intrins (3, 3)
    intrins = None
    if args.load_intrins:
        intrins = np.load(intrins_path)

    sample = Sample(
        img=img,
        normal=normal,
        normal_mask=normal_mask,
        intrins=intrins,

        dataset_name='scannet',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample
