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
    parser.add_argument('--NNET_architecture', type=str, default='v02')
    parser.add_argument('--NNET_output_dim', type=int, default=3, help='{3, 4}')
    parser.add_argument('--NNET_output_type', type=str, default='R', help='{R, G}')
    parser.add_argument('--NNET_feature_dim', type=int, default=64)
    parser.add_argument('--NNET_hidden_dim', type=int, default=64)

    parser.add_argument('--NNET_encoder_B', type=int, default=5)

    parser.add_argument('--NNET_decoder_NF', type=int, default=2048)
    parser.add_argument('--NNET_decoder_BN', default=False, action="store_true")
    parser.add_argument('--NNET_decoder_down', type=int, default=8)
    parser.add_argument('--NNET_learned_upsampling', default=False, action="store_true")

    parser.add_argument('--NRN_prop_ps', type=int, default=5)
    parser.add_argument('--NRN_num_iter_train', type=int, default=5)
    parser.add_argument('--NRN_num_iter_test', type=int, default=5)
    parser.add_argument('--NRN_ray_relu', default=False, action="store_true")

    parser.add_argument('--loss_fn', type=str, default='AL')
    parser.add_argument('--loss_gamma', type=float, default=0.8)
    #↑↑↑↑

    # read arguments from txt file
    assert '.txt' in sys.argv[1]
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix] + sys.argv[2:])

    #↓↓↓↓
    #NOTE: update args
    args.exp_root = os.path.join(EXPERIMENT_DIR, 'dsine')
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

