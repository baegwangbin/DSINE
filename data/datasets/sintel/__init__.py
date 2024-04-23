""" Get samples from Sintel (http://sintel.is.tue.mpg.de/)
    NOTE: We computed the GT surface normals by doing discontinuity-aware plane fitting
"""
import os
import cv2
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from data import Sample

from projects import DATASET_DIR
DATASET_PATH = os.path.join(DATASET_DIR, 'dsine_eval', 'sintel')


def get_sample(args, sample_path, info):
    # e.g. sample_path = "alley_1/frame_0001_img.png"
    scene_name = sample_path.split('/')[0]
    img_name, img_ext = sample_path.split('/')[1].split('_img')

    img_path = '%s/%s' % (DATASET_PATH, sample_path)
    normal_path = img_path.replace('_img'+img_ext, '_normal.exr')
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
        normal_mask = np.linalg.norm(normal, axis=2, keepdims=True) > 0.5

    # read intrins (3, 3)
    intrins = None
    if args.load_intrins:
        intrins = np.load(intrins_path)

    sample = Sample(
        img=img,
        normal=normal,
        normal_mask=normal_mask,
        intrins=intrins,

        dataset_name='sintel',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample