import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

from utils.projection import intrins_from_fov
from utils.utils import get_padding


def define_input(input, device='cpu', **kwargs):
    if input == 'screen':
        InputStream = InputScreen(device=device, **kwargs)
    elif input == 'webcam':
        InputStream = InputWebcam(device=device, **kwargs)
    elif input == 'rs':
        InputStream = InputRealsense(device=device, **kwargs) 
    elif input == 'youtube':
        InputStream = InputYoutube(device=device, **kwargs) 
    elif input == 'video':
        InputStream = InputVideo(device=device, **kwargs) 
    else:
        raise Exception('input option %s is not valid' % input)
    return InputStream


def img_to_tensor(color_image, lrtb=(0,0,0,0), device="cpu"):
    """ color_image is a BGR numpy array of shape (H, W, 3)
        this function returns an RGB torch tensor of shape (1, H, W, 3)
    """
    # NOTE: THIS IS SLOW
    # img = color_image.astype(np.float32) / 255.0
    # img = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0)

    # NOTE: THIS IS FASTER
    img = color_image[:, :, ::-1].astype(np.uint8)
    img = torch.from_numpy(img).to(device).permute(2, 0, 1).to(dtype=torch.float32)
    img = (img / 255.0).unsqueeze(0).contiguous()
    img = F.pad(img, lrtb, mode="constant", value=0.0)
    return img


class InputScreen():
    def __init__(self, device=0, 
                 intrins: torch.Tensor = None,
                 top: int = (1080-480) // 2,
                 left: int = (1920-640) // 2,
                 height: int = 480,
                 width: int = 640,
                 **kwargs
                 ):
        from mss import mss

        self.device = device
        self.bounding_box = {'top': top, 'left': left, 'width': width, 'height': height}
        self.sct = mss()

        # input should be padded to ensure that both H and W are divisible by 32
        self.new_H, self.new_W = height, width
        self.lrtb = get_padding(self.new_H, self.new_W)

        # intrins (if None, assume that FoV is 60)
        # intrins should be updated after padding
        if intrins is None:
            self.intrins = intrins_from_fov(new_fov=60.0, H=self.new_H, W=self.new_W, device=device)
        else:
            self.intrins = intrins

        if self.intrins.ndim == 2:
            self.intrins = self.intrins.unsqueeze(0)
        self.intrins[:, 0, 2] += self.lrtb[0]
        self.intrins[:, 1, 2] += self.lrtb[2]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_sample(self):
        # color_image: numpy array, BGR, (H, W, 3)
        color_image = np.array(self.sct.grab(self.bounding_box))[:,:,:3]

        # img: torch tensor, RGB, (1, 3, H, W)
        img = img_to_tensor(color_image, lrtb=self.lrtb, device=self.device)
        img = self.normalize(img)

        # intrins: torch tensor (3, 3)
        intrins = self.intrins.clone()

        # sample
        sample = {
            'color_image': color_image,
            'img': img,
            'intrins': intrins,
        }

        return sample


class InputWebcam():
    def __init__(self, device=0, 
                 intrins: torch.Tensor = None,
                 new_width: int = -1,
                 webcam_index: int = 1,
                 **kwargs
                 ):
        self.device = device
        self.cap = cv2.VideoCapture(webcam_index)
        assert self.cap.isOpened()

        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                first_frame = frame
                break

        # input should be padded to ensure that both H and W are divisible by 32
        self.orig_H, self.orig_W, _ = first_frame.shape
        if new_width == -1:
            self.new_H, self.new_W = self.orig_H, self.orig_W
            self.interp = None
        else:
            self.new_W = new_width
            self.new_H = round(self.orig_H * (self.new_W / self.orig_W))
            if self.new_W < self.orig_W:
                self.interp = cv2.INTER_AREA
            else:
                self.interp = cv2.INTER_LINEAR
        self.lrtb = get_padding(self.new_H, self.new_W)

        # intrins (if None, assume that FoV is 60)
        # intrins should be updated after padding
        if intrins is None:
            self.intrins = intrins_from_fov(new_fov=60.0, H=self.new_H, W=self.new_W, device=device)
        else:
            self.intrins = intrins

        if self.intrins.ndim == 2:
            self.intrins = self.intrins.unsqueeze(0)
        self.intrins[:, 0, 2] += self.lrtb[0]
        self.intrins[:, 1, 2] += self.lrtb[2]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_sample(self):
        # color_image: numpy array, BGR, (H, W, 3)
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                color_image = frame
                break

        if self.interp is not None:
            color_image = cv2.resize(color_image, (self.new_W, self.new_H), interpolation=self.interp)

        # img: torch tensor, RGB, (1, 3, H, W)
        img = img_to_tensor(color_image, lrtb=self.lrtb, device=self.device)
        img = self.normalize(img)

        # intrins: torch tensor (3, 3)
        intrins = self.intrins.clone()

        # sample
        sample = {
            'color_image': color_image,
            'img': img,
            'intrins': intrins,
        }

        return sample


class InputRealsense():
    def __init__(self, device=0, 
                 enable_auto_exposure: bool = False, 
                 enable_auto_white_balance: bool = False,
                 **kwargs
                 ):
        import pyrealsense2 as rs

        self.device = device
        # load camera
        # Configure depth and color streams
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.rs_pipeline)
        pipeline_profile = self.rs_config.resolve(pipeline_wrapper)
        rs_device = pipeline_profile.get_device()

        found_rgb = False
        for s in rs_device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                s.set_option(rs.option.enable_auto_exposure, enable_auto_exposure)
                s.set_option(rs.option.enable_auto_white_balance, enable_auto_white_balance)
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.rs_pipeline.start(self.rs_config)
        profile = self.rs_pipeline.get_active_profile()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # intrins - fixed for realsense
        self.new_H, self.new_W = intr.height, intr.width
        self.lrtb = get_padding(self.new_H, self.new_W)

        fx = intr.fx
        cx = intr.ppx
        fy = intr.fy
        cy = intr.ppy

        self.intrins = torch.tensor([[fx,  0.0, cx ],
                                     [0.0, fy,  cy ],
                                     [0.0, 0.0, 1.0]], dtype=torch.float32, device=device).unsqueeze(0)
        self.intrins[:, 0, 2] += self.lrtb[0]
        self.intrins[:, 1, 2] += self.lrtb[2]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_sample(self):
        # color_image: numpy array, BGR, (H, W, 3)
        frames = self.rs_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # img: torch tensor, RGB, (1, 3, H, W)
        img = img_to_tensor(color_image, lrtb=self.lrtb, device=self.device)
        img = self.normalize(img)

        # intrins: torch tensor (3, 3)
        intrins = self.intrins.clone()

        # sample
        sample = {
            'color_image': color_image,
            'img': img,
            'intrins': intrins,
        }

        return sample


class InputYoutube():
    def __init__(self, device=0,
                 intrins: torch.Tensor = None,
                 new_width: int = -1,
                 video_id: str = 'dQw4w9WgXcQ', 
                 **kwargs):
        self.device = device
        self.video_id = video_id

        # read first image and set intrins
        self.init_stream()
        first_frame = self.stream.read()

        # input should be padded to ensure that both H and W are divisible by 32
        self.orig_H, self.orig_W, _ = first_frame.shape
        if new_width == -1:
            self.new_H, self.new_W = self.orig_H, self.orig_W
            self.interp = None
        else:
            self.new_W = new_width
            self.new_H = round(self.orig_H * (self.new_W / self.orig_W))
            if self.new_W < self.orig_W:
                self.interp = cv2.INTER_AREA
            else:
                self.interp = cv2.INTER_LINEAR
        self.lrtb = get_padding(self.new_H, self.new_W)

        # intrins (if None, assume that FoV is 60)
        # intrins should be updated after padding
        if intrins is None:
            self.intrins = intrins_from_fov(new_fov=60.0, H=self.new_H, W=self.new_W, device=device)
        else:
            self.intrins = intrins

        if self.intrins.ndim == 2:
            self.intrins = self.intrins.unsqueeze(0)
        self.intrins[:, 0, 2] += self.lrtb[0]
        self.intrins[:, 1, 2] += self.lrtb[2]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.init_stream()

    def init_stream(self):
        from vidgear.gears import CamGear
        self.stream = CamGear(source='https://youtu.be/%s' % self.video_id, stream_mode=True, logging=True).start()

    def get_sample(self):
        # color_image: numpy array, BGR, (H, W, 3)
        color_image = self.stream.read()
        if color_image is None:
            self.init_stream()
            color_image = self.stream.read()

        if self.interp is not None:
            color_image = cv2.resize(color_image, (self.new_W, self.new_H), interpolation=self.interp)

        # img: torch tensor, RGB, (1, 3, H, W)
        img = img_to_tensor(color_image, lrtb=self.lrtb, device=self.device)
        img = self.normalize(img)

        # intrins: torch tensor (3, 3)
        intrins = self.intrins.clone()

        # sample
        sample = {
            'color_image': color_image,
            'img': img,
            'intrins': intrins,
        }

        return sample


class InputVideo():
    def __init__(self, device=0, 
                 intrins: torch.Tensor = None,
                 new_width: int = -1,
                 video_path: str = None,
                 **kwargs
                 ):
        self.device = device
        assert os.path.exists(video_path)
        self.video_path = video_path
        self.vidcap = cv2.VideoCapture(self.video_path)
        _, first_frame = self.vidcap.read()

        # input should be padded to ensure that both H and W are divisible by 32
        self.orig_H, self.orig_W, _ = first_frame.shape
        if new_width == -1:
            self.new_H, self.new_W = self.orig_H, self.orig_W
            self.interp = None
        else:
            self.new_W = new_width
            self.new_H = round(self.orig_H * (self.new_W / self.orig_W))
            if self.new_W < self.orig_W:
                self.interp = cv2.INTER_AREA
            else:
                self.interp = cv2.INTER_LINEAR
        self.lrtb = get_padding(self.new_H, self.new_W)

        # intrins (if None, assume that FoV is 60)
        # intrins should be updated after padding
        if intrins is None:
            self.intrins = intrins_from_fov(new_fov=60.0, H=self.new_H, W=self.new_W, device=device)
        else:
            self.intrins = intrins

        if self.intrins.ndim == 2:
            self.intrins = self.intrins.unsqueeze(0)
        self.intrins[:, 0, 2] += self.lrtb[0]
        self.intrins[:, 1, 2] += self.lrtb[2]

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_sample(self):
        # color_image: numpy array, BGR, (H, W, 3)
        success, color_image = self.vidcap.read()
        if not success:
            self.vidcap = cv2.VideoCapture(self.video_path)
            success, color_image = self.vidcap.read()
            assert success            

        if self.interp is not None:
            color_image = cv2.resize(color_image, (self.new_W, self.new_H), interpolation=self.interp)

        # img: torch tensor, RGB, (1, 3, H, W)
        img = img_to_tensor(color_image, lrtb=self.lrtb, device=self.device)
        img = self.normalize(img)

        # intrins: torch tensor (3, 3)
        intrins = self.intrins.clone()

        # sample
        sample = {
            'color_image': color_image,
            'img': img,
            'intrins': intrins,
        }

        return sample


