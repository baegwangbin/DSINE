import os
import sys
import numpy as np
from tqdm import tqdm
import glob

import torch
import torch.nn.functional as F

import time
from torchvision import transforms
import cv2
from PIL import Image

import sys
sys.path.append('../../')
import utils.utils as utils
import utils.visualize as vis_utils

#↓↓↓↓
#NOTE: project-specific imports (e.g. config)
import projects.baseline_normal.config as config
from projects.baseline_normal.dataloader import *
#↑↑↑↑


def test(args, model, test_loader, device, results_dir=None):
    with torch.no_grad():
        total_normal_errors = None

        for data_dict in tqdm(test_loader):

            #↓↓↓↓
            #NOTE: forward pass
            img = data_dict['img'].to(device)
            scene_names = data_dict['scene_name']
            img_names = data_dict['img_name']

            # pad input
            _, _, orig_H, orig_W = img.shape
            lrtb = utils.get_padding(orig_H, orig_W)
            img, _ = utils.pad_input(img, None, lrtb)

            # forward pass
            norm_out = model(img)

            # crop the padded part
            norm_out = norm_out[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

            pred_norm, pred_kappa = norm_out[:, :3, :, :], norm_out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa
            #↑↑↑↑

            if 'normal' in data_dict.keys():
                gt_norm = data_dict['normal'].to(device)
                gt_norm_mask = data_dict['normal_mask'].to(device)

                pred_error = utils.compute_normal_error(pred_norm, gt_norm)
                if total_normal_errors is None:
                    total_normal_errors = pred_error[gt_norm_mask]
                else:
                    total_normal_errors = torch.cat((total_normal_errors, pred_error[gt_norm_mask]), dim=0)

            if results_dir is not None:
                prefixs = ['%s_%s' % (i,j) for (i,j) in zip(scene_names, img_names)]
                vis_utils.visualize_normal(results_dir, prefixs, img, pred_norm, pred_kappa,
                                           gt_norm, gt_norm_mask, pred_error)

        if total_normal_errors is not None:
            metrics = utils.compute_normal_metrics(total_normal_errors)
            print("mean median rmse 5 7.5 11.25 22.5 30")
            print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
                metrics['mean'], metrics['median'], metrics['rmse'],
                metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))


def test_samples(args, model, device):
    img_paths = glob.glob('./samples/img/*.png') + glob.glob('./samples/img/*.jpg')
    img_paths.sort()
    os.makedirs('./samples/output/', exist_ok=True)

    # normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for img_path in img_paths:
            print(img_path)
            ext = os.path.splitext(img_path)[1]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            #↓↓↓↓
            #NOTE: forward pass

            # pad input
            _, _, orig_H, orig_W = img.shape
            lrtb = utils.get_padding(orig_H, orig_W)
            img = F.pad(img, lrtb, mode="constant", value=0.0)
            img = normalize(img)

            norm_out = model(img)
            norm_out = norm_out[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]
            pred_norm = norm_out[:, :3, :, :]
            #↑↑↑↑

            # save to output folder
            # by saving the prediction as uint8 png format, you lose a lot of precision
            # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files
            target_path = img_path.replace('/img/', '/output/').replace(ext, '.png')
            im = Image.fromarray(vis_utils.normal_to_rgb(pred_norm)[0,...])
            im.save(target_path)


def measure_throughput(model, img, dtype='fp32', nwarmup=50, nruns=1000):
    img = img.to("cuda")
    if dtype=='fp16':
        img = img.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            norm_out = model(img)

    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            norm_out = model(img)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", img.size())
    print('Average throughput: %.2f images/second'%(img.shape[0]/np.mean(timings)))


def demo(args, model, InputStream, frame_name):
    cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(frame_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    pause = False

    lrtb = InputStream.lrtb
    H, W = InputStream.new_H, InputStream.new_W

    while True:
        with torch.no_grad():
            if pause:
                pass
            else:
                data_dict = InputStream.get_sample()
                color_image = data_dict['color_image']

                #↓↓↓↓
                #NOTE: forward pass

                img = data_dict['img']

                norm_out = model(img)

                norm_out = norm_out[:, :, lrtb[2]:lrtb[2]+H, lrtb[0]:lrtb[0]+W]
                pred_norm = norm_out[:, :3, :, :]
                pred_kappa = norm_out[:, 3:, :, :]    
                #↑↑↑↑

                # visualize
                pred_norm_rgb = vis_utils.normal_to_rgb(pred_norm)[0,...][...,::-1]
                if pred_kappa.size(1) == 0:
                    pred_uncertainty = []
                elif 'NLL_angmf' in args.loss_fn:
                    pred_uncertainty = [vis_utils.alpha_to_jet(vis_utils.kappa_to_alpha(pred_kappa))] # BGR
                else:
                    pred_uncertainty = [vis_utils.depth_to_rgb(pred_kappa[0,...], None, 0.0, None, 'gray')[...,::-1]]
                out = np.hstack([color_image, pred_norm_rgb]+pred_uncertainty)

                cv2.imshow(frame_name, out)

            # keyboard input
            k = cv2.waitKey(1)
            if k == ord(' '):
                pause = not pause
            elif k == ord('q'):
                exit()


if __name__ == '__main__':
    device = torch.device('cuda')
    args = config.get_args(test=True)

    if args.ckpt_path is None:
        ckpt_paths = glob.glob(os.path.join(args.output_dir, 'models', '*.pt'))
        ckpt_paths.sort()
        args.ckpt_path = ckpt_paths[-1]
    assert os.path.exists(args.ckpt_path)

    #↓↓↓↓
    #NOTE: define and load model
    from projects.baseline_normal.model import NNET
    model = NNET(args).to(device)

    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()
    #↑↑↑↑
 
    # test the model
    if args.mode == 'benchmark':
        # do not resize/crop the images when benchmarking
        args.input_height = args.input_width = 0
        args.data_augmentation_same_fov = 0

        for dataset_name, split in [('nyuv2', 'test'), 
                                    ('scannet', 'test'),
                                    ('ibims', 'ibims'),
                                    ('sintel', 'sintel'),
                                    ('vkitti', 'vkitti'),
                                    ('oasis', 'val')
                                    ]:

            args.dataset_name_test = dataset_name
            args.test_split = split
            test_loader = TestLoader(args).data

            results_dir = None
            if args.visualize:
                results_dir = os.path.join(args.output_dir, 'test', dataset_name)
                os.makedirs(results_dir, exist_ok=True)
            
            test(args, model, test_loader, device, results_dir)

    # test on samples
    elif args.mode == 'samples':
        test_samples(args, model, device)

    #↓↓↓↓
    #NOTE: measure throughput
    elif args.mode == 'throughput':
        H, W = 480, 640
        batch_size = 8
        dummy_img = torch.rand(batch_size, 3, H, W).float().to(device)
        measure_throughput(model, dummy_img, dtype='fp32')
    #↑↑↑↑

    # demo
    else:
        from utils.demo_data import define_input
        if args.mode == 'screen':
            input_name = 'screen'
            kwargs = dict(
                intrins = None,
                top = (1080-480) // 2,
                left = (1920-640) // 2,
                height = 480,
                width = 640,
            )

        elif args.mode == 'webcam':
            input_name = 'webcam'
            kwargs = dict(
                intrins = None,
                new_width = -1,
                webcam_index = 1,
            )

        elif args.mode == 'rs':
            input_name = 'rs'
            kwargs = dict(
                enable_auto_exposure = True,
                enable_auto_white_balance = True,
            )

        elif 'youtube.com' in args.mode:
            input_name = 'youtube'
            kwargs = dict(
                intrins = None,
                new_width = 1024,
                video_id = args.mode.split('watch?v=')[1],
            )

        else:
            raise Exception('invalid input option for demo')

        InputStream = define_input(input=input_name, device=device, **kwargs)
        demo(args, model, InputStream, frame_name=args.ckpt_path)


