""" Get samples from OASIS validation set (https://pvl.cs.princeton.edu/OASIS/)
"""
import os
import cv2
import numpy as np
import pickle

from data import Sample

from projects import DATASET_DIR
DATASET_PATH = os.path.join(DATASET_DIR, 'dsine_eval', 'oasis')


def read_normal(path, h, w):
    normal_dict = pickle.load(open(path, 'rb'))

    mask = np.zeros((h,w))
    normal = np.zeros((h,w,3))

    # Stuff ROI normal into bounding box
    min_y = normal_dict['min_y']
    max_y = normal_dict['max_y']
    min_x = normal_dict['min_x']
    max_x = normal_dict['max_x']
    roi_normal = normal_dict['normal']

    # to LUB
    normal[min_y:max_y+1, min_x:max_x+1, :] = roi_normal
    normal = normal.astype(np.float32)
    normal[:,:,0] *= -1
    normal[:,:,1] *= -1

    # Make mask
    roi_mask = np.logical_or(np.logical_or(roi_normal[:,:,0] != 0, roi_normal[:,:,1] != 0), roi_normal[:,:,2] != 0).astype(np.float32)
    mask[min_y:max_y+1, min_x:max_x+1] = roi_mask
    mask = mask[:, :, None]
    mask = mask > 0.5

    return normal, mask


def get_sample(args, sample_path, info):
    # e.g. sample_path = "val/100277_DT_img.png"
    scene_name = sample_path.split('/')[0]
    img_name, img_ext = sample_path.split('/')[-1].split('_img')

    img_path = '%s/%s' % (DATASET_PATH, sample_path)
    normal_path = img_path.replace('_img'+img_ext, '_normal.pkl')
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
        h = img.shape[0]
        w = img.shape[1]
        normal, normal_mask = read_normal(normal_path, h, w)

    # read intrins (3, 3)
    intrins = None
    if args.load_intrins:
        intrins = np.load(intrins_path)

    sample = Sample(
        img=img,
        normal=normal,
        normal_mask=normal_mask,
        intrins=intrins,

        dataset_name='oasis',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample
