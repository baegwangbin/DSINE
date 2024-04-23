""" Arguments that are shared by all projects
"""
import os
import argparse

DATASET_DIR = '/media/gwangbin/ssd/datasets/'   # where datasets are stored         NOTE: this should be changed!
EXPERIMENT_DIR = '/home/gwangbin/experiments/'  # where to save the experiments     NOTE: this should be changed!
PROJECT_DIR = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]


def get_default_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # experiment path
    parser.add_argument('--exp_root', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='exp_name')
    parser.add_argument('--exp_id', type=str, default='exp_id')

    # training
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, help='accumulate gradient every N batches')
    parser.add_argument("--workers", default=12, type=int)

    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument('--gpus', type=str, default='-1', help='which gpus to use, if -1, use all')
    parser.add_argument('--save_all_models', action='store_true')
    parser.add_argument('--overwrite_models', action='store_true', help='if True, overwrite the existing checkpoints')

    parser.add_argument('--lr', default=0.0003, type=float, help='max learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--diff_lr', action="store_true", help="use different LR for different network components")
    parser.add_argument('--grad_clip', default=1.0, type=float)

    parser.add_argument('--validate_every', default=1e20, type=int, help='validate every N iterations, validation also happens every epoch')
    parser.add_argument('--visualize_every', default=1000, type=int, help='visualize every N iterations')

    # dataset - what to load
    parser.add_argument("--load_img", default=True, action="store_true")
    parser.add_argument("--load_depth", default=False, action="store_true")
    parser.add_argument("--load_normal", default=False, action="store_true")
    parser.add_argument("--load_intrins", default=True, action="store_true")

    # dataset - preprocessing
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)

    # dataset - names and split
    parser.add_argument("--dataset_name_train", type=str, default='nyuv2')
    parser.add_argument("--dataset_name_val", type=str, default='nyuv2')
    parser.add_argument("--dataset_name_test", type=str, default=None)

    parser.add_argument("--train_split", type=str, default='train')
    parser.add_argument("--val_split", type=str, default='test')
    parser.add_argument("--test_split", type=str, default=None)

    # dataset - preprocessing & augmentation
    # for more information, refer to the functions in data/augmentation
    parser.add_argument("--data_augmentation_intrins", default=False, action="store_true")
    parser.add_argument("--data_augmentation_same_fov", type=int, default=0)

    parser.add_argument("--data_augmentation_persp", default=False, action="store_true")
    parser.add_argument("--data_augmentation_persp_yaw", type=int, default=0)
    parser.add_argument("--data_augmentation_persp_pitch", type=int, default=0)
    parser.add_argument("--data_augmentation_persp_roll", type=int, default=0)
    parser.add_argument("--data_augmentation_persp_random_fov", default=False, action="store_true")
    parser.add_argument("--data_augmentation_persp_min_fov", type=float, default=60.0)
    parser.add_argument("--data_augmentation_persp_max_fov", type=float, default=90.0)

    parser.add_argument("--data_augmentation_hflip", default=False, action="store_true")

    parser.add_argument("--data_augmentation_crop", default=False, action="store_true")
    parser.add_argument("--data_augmentation_crop_height", type=int, default=416)
    parser.add_argument("--data_augmentation_crop_width", type=int, default=544)

    parser.add_argument("--data_augmentation_color", default=False, action="store_true")
    
    parser.add_argument("--data_augmentation_appear", type=int, default=0)

    parser.add_argument("--data_augmentation_nyu_crop", default=False, action="store_true")

    # checkpoint (only needed when testing the model)
    parser.add_argument('--ckpt_path', type=str, default=None)

    # arguments for testing
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--visualize', default=False, action="store_true")

    return parser


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)