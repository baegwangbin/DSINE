import os
import sys
import glob
from datetime import datetime

from projects import DATASET_DIR, EXPERIMENT_DIR, get_default_parser
import utils.utils as utils

import logging
logger = logging.getLogger('root')


def get_args(test=False):
    parser = get_default_parser()

    #↓↓↓↓
    #NOTE: project-specific args
    parser.add_argument('--NNET_architecture', type=str, default='densedepth__B5__NF2048__down2__bilinear__BN')
    parser.add_argument('--NNET_output_dim', type=int, default=4, help='{3, 4}')
    parser.add_argument('--NNET_output_type', type=str, default='G', help='{R, G}')

    parser.add_argument('--loss_fn', type=str, default='NLL_angmf')
    #↑↑↑↑

    # read arguments from txt file
    assert '.txt' in sys.argv[1]
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix] + sys.argv[2:])

    #↓↓↓↓
    #NOTE: update args
    args.exp_root = os.path.join(EXPERIMENT_DIR, 'baseline_normal')
    args.load_normal = True
    args.load_intrins = True
    #↑↑↑↑

    # set working dir
    exp_dir = os.path.join(args.exp_root, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    args.output_dir = os.path.join(exp_dir, args.exp_id)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'log'), exist_ok=True)        # save log
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)     # save models
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)       # save test images

    if not test and \
        not args.overwrite_models and \
            len(glob.glob(os.path.join(args.output_dir, 'models', '*.pt'))) > 0:
        print('checkpoints exist!')
        exit()

    # training
    if not test:
        global logger
        utils.change_logger_dest(logger, os.path.join(args.output_dir, 'log', '%s.log' % datetime.now()))
        from torch.utils.tensorboard.writer import SummaryWriter
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'log'))

    # save args
    args_path = os.path.join(args.output_dir, 'log', 'params.txt')
    utils.save_args(args, args_path)
    logger.info('config saved in %s' % args_path)

    # log
    logger.info('DATASET_DIR: %s' % DATASET_DIR)
    logger.info('EXPERIMENT_DIR: %s' % args.output_dir)

    return args

