import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data.distributed

import sys
sys.path.append('../../')
import utils.utils as utils
import utils.visualize as vis_utils

# setup logging
logger = utils.setup_custom_logger('root')
import logging
logging.getLogger('PIL').setLevel(logging.INFO)

#↓↓↓↓
#NOTE: project-specific imports (e.g. config)
import projects.baseline_normal.config as config
from projects.baseline_normal.dataloader import *
from projects.baseline_normal.losses import ComputeLoss
#↑↑↑↑

BEST_KEY = 'mean'   # metric to use when selecting the best model


def train(model, args, device):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    should_write = ((not args.distributed) or args.rank == 0)
    if should_write:
        logger.info('Number of model parameters: %s' % sum([p.data.nelement() for p in model.parameters()]))

    # train & val dataloader
    train_loader = TrainLoader(args, epoch=0).data
    val_loader = ValLoader(args).data

    # define losses
    loss_fn = ComputeLoss(args)

    # optimizer
    if not args.diff_lr:
        logger.info("Using same LR")
        params = model.parameters()
    else:
        logger.info("Using diff LR")
        m = model.module if args.multigpu else model
        #↓↓↓↓
        #NOTE: For some parameters (e.g. those in encoder), we use 1/10 learning rate. This part may need to be updated depending on how you defined your model.
        params = [{"params": m.n_net.model.get_1x_lr_params(), "lr": args.lr / 10},
                  {"params": m.n_net.model.get_10x_lr_params(), "lr": args.lr}]
        #↑↑↑↑
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay, lr=args.lr)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              max_lr=args.lr,
                                              epochs=args.num_epochs,
                                              steps_per_epoch=len(train_loader) // args.accumulate_grad_batches,
                                              div_factor=25.0,
                                              final_div_factor=10000.0)

    # cudnn setting
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    # best accuracy (lower the better)
    best_acc = 1e4

    # start training
    total_iter = 0
    model.train()
    for epoch in range(args.num_epochs):
        train_loader = TrainLoader(args, epoch=epoch).data

        if args.rank == 0:
            data_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.num_epochs}. Loop: Train")
        else:
            data_loader = train_loader

        for batch_idx, data_dict in enumerate(data_loader):
            total_iter += args.batch_size_orig

            #↓↓↓↓
            #NOTE: forward pass
            img = data_dict['img'].to(device)
            gt_norm = data_dict['normal'].to(device)
            gt_norm_mask = data_dict['normal_mask'].to(device)

            norm_out = model(img)
            loss = loss_fn(norm_out, gt_norm, gt_norm_mask)
            #↑↑↑↑

            # back-propagate
            loss = loss / args.accumulate_grad_batches
            scaler.scale(loss).backward()

            if ((batch_idx + 1) % args.accumulate_grad_batches == 0):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # log loss
            if should_write:
                loss_ = float(loss.data.cpu().numpy())
                args.writer.add_scalar('loss', loss_, global_step=total_iter)
                data_loader.set_description(f"Epoch: {epoch + 1}/{args.num_epochs}. Loop: Train. Loss: {'%.5f' % loss_}")
                data_loader.refresh()

            #↓↓↓↓
            #NOTE: visualize (in tensorboard)
            if should_write and ((total_iter % args.visualize_every) < args.batch_size_orig):
                vis_ = vis_utils.visualize_normal_tb(args, img, norm_out, gt_norm, gt_norm_mask)
                args.writer.add_image('train vis', vis_, global_step=total_iter, dataformats='HWC')
            #↑↑↑↑

            # validation
            if should_write and ((total_iter % args.validate_every) < args.batch_size_orig):
                if args.save_all_models:
                    utils.save_model(model, os.path.join(args.output_dir, 'models', 'iter_%010d.pt' % total_iter), total_iter)
                else:
                    model.eval()
                    metrics = validate(model, args, val_loader, device, total_iter)
                    if metrics[BEST_KEY] <= best_acc:
                        utils.save_model(model, os.path.join(args.output_dir, 'models', 'best.pt'), total_iter)
                        best_acc = metrics[BEST_KEY]
                        print('best acc: %s' % best_acc)
                    model.train()

        # validation after epoch
        if should_write:
            if args.save_all_models:
                utils.save_model(model, os.path.join(args.output_dir, 'models', 'iter_%010d.pt' % total_iter), total_iter)
            else:
                model.eval()
                metrics = validate(model, args, val_loader, device, total_iter)
                if metrics[BEST_KEY] <= best_acc:
                    utils.save_model(model, os.path.join(args.output_dir, 'models', 'best.pt'), total_iter)
                    best_acc = metrics[BEST_KEY]
                    print('best acc: %s' % best_acc)
                if epoch+1 == args.num_epochs:
                    utils.save_model(model, os.path.join(args.output_dir, 'models', 'last.pt'), total_iter)
                model.train()

        del train_loader
    return model

    
def validate(model, args, val_loader, device, total_iter):
    with torch.no_grad():
        total_metrics = utils.RunningAverageDict()
        for data_dict in tqdm(val_loader, desc="Loop: Validation"):

            #↓↓↓↓
            #NOTE: forward pass
            img = data_dict['img'].to(device)
            gt_norm = data_dict['normal'].to(device)
            gt_norm_mask = data_dict['normal_mask'].to(device)

            norm_out = model(img)
            pred_norm = norm_out[:, :3, :, :]
            #↑↑↑↑

            pred_error = utils.compute_normal_error(pred_norm, gt_norm)
            metrics_, num_pixels = utils.compute_normal_metrics2(pred_error[gt_norm_mask])
            total_metrics.update(metrics_, num_pixels)

        metrics = total_metrics.get_value()
        metrics['rmse'] = np.sqrt(metrics['mse'])

        # tb logging
        for k, v in metrics.items():
            args.writer.add_scalar(k, v, global_step=total_iter)
        return metrics


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    #↓↓↓↓
    #NOTE: define model
    from projects.baseline_normal.model import NNET
    model = NNET(args)
    #↑↑↑↑

    # If a gpu is set by user: NO PARALLELISM
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

        logger.info('GPU: %s / RANK: %s / Batch size: %s / Num workers: %s' %
              (args.gpu, args.rank, args.batch_size, args.workers))

        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    train(model, args, device=args.gpu)


if __name__ == '__main__':
    args = config.get_args()

    # set visible gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(args.gpus))

    args.world_size = 1
    args.rank = 0
    nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    args.batch_size_orig = args.batch_size

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
