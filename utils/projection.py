import numpy as np
import torch
import torch.nn.functional as F


def intrins_zero_to(intrins):
    """ add 0.5 to ensure top-left pixel is (0.5, 0.5)

        NOTE: intrins should have the shape (..., 3, 3)
    """
    intrins[..., 0, 2] += 0.5
    intrins[..., 1, 2] += 0.5
    return intrins


def intrins_to_zero(intrins):
    """ subtract 0.5 to ensure top-left pixel is (0, 0)

        NOTE: intrins should have the shape (..., 3, 3)
    """
    intrins[..., 0, 2] -= 0.5
    intrins[..., 1, 2] -= 0.5
    return intrins


def intrins_crop(intrins, 
                 crop_top: int = 0, 
                 crop_left: int = 0):
    """ update intrins after crop

        NOTE: intrins should have the shape (..., 3, 3)
    """
    intrins[..., 0, 2] -= crop_left
    intrins[..., 1, 2] -= crop_top
    return intrins


def intrins_resize(intrins, 
                   ratio_H: float = 1.0, 
                   ratio_W: float = 1.0):
    """ update intrins after resize

        NOTE: intrins should have the shape (..., 3, 3)
        NOTE: top-left is (0,0)
    """
    intrins = intrins_zero_to(intrins)
    intrins[..., 0, 0] *= ratio_W   # fx
    intrins[..., 0, 2] *= ratio_W   # cx
    intrins[..., 1, 1] *= ratio_H   # fy
    intrins[..., 1, 2] *= ratio_H   # cy
    intrins = intrins_to_zero(intrins)
    return intrins


def get_intrins(fx, fy, cx, cy, dtype=torch.float32, device='cpu'):
    """ intrins from fx, fy, cx, cy

        NOTE: top-left is (0,0)
    """
    intrins = torch.tensor([[ fx, 0.0,  cx],
                            [0.0,  fy,  cy],
                            [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    intrins_inv = torch.tensor([[1/fx,  0.0, -cx/fx],
                                [ 0.0, 1/fy, -cy/fy],
                                [ 0.0,  0.0,    1.0]], dtype=dtype, device=device)
    return intrins, intrins_inv


def intrins_to_intrins_inv(intrins):
    """ intrins to intrins_inv

        NOTE: top-left is (0,0)
    """
    if torch.is_tensor(intrins):
        intrins_inv = torch.zeros_like(intrins)
    elif type(intrins) is np.ndarray:
        intrins_inv = np.zeros_like(intrins)
    else:
        raise Exception('intrins should be torch tensor or numpy array')

    intrins_inv[0, 0] = 1 / intrins[0, 0]
    intrins_inv[0, 2] = - intrins[0, 2] / intrins[0, 0]
    intrins_inv[1, 1] = 1 / intrins[1, 1]
    intrins_inv[1, 2] = - intrins[1, 2] / intrins[1, 1]
    intrins_inv[2, 2] = 1.0
    return intrins_inv


def intrins_from_fov(new_fov, H, W, dtype=torch.float32, device='cpu'):
    """ define intrins based on field-of-view
        principal point is assumed to be at the center

        NOTE: new_fov should be in degrees
        NOTE: top-left is (0,0)
    """
    new_fx = new_fy = (max(H, W) / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    new_cx = (W / 2.0) - 0.5
    new_cy = (H / 2.0) - 0.5

    new_intrins = torch.tensor([
        [new_fx,    0,          new_cx  ],
        [0,         new_fy,     new_cy  ],
        [0,         0,          1       ]
    ], dtype=dtype, device=device)

    return new_intrins


def intrins_from_fov2(new_fov, H, W, cx, cy, dtype=torch.float32, device='cpu'):
    """ define intrins based on field-of-view
        principal point is assumed to be at (cx, cy) 

        NOTE: new_fov should be in degrees
        NOTE: top-left is (0,0)
    """    
    cx += 0.5
    cy += 0.5

    if W >= H:
        x1 = W - cx
        x2 = cx
    else:
        x1 = H - cy
        x2 = cy

    # use tan(x+y) = (tan(x) + tan(y)) / (1 – tan(x)tan(y))
    A = np.tan(np.deg2rad(new_fov))
    B = - (x1 + x2)
    C = - np.tan(np.deg2rad(new_fov)) * x1 * x2
    new_f = (-B + np.sqrt(B**2.0 - (4 * A * C))) / (2*A)

    intrins = torch.tensor([[new_f, 0.0,  cx-0.5],
                            [0.0,  new_f, cy-0.5],
                            [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    return intrins


def intrins_from_txt(intrins_path, dtype=torch.float32, device='cpu'):
    """ define intrins based on txt
        it should contain fx,fy,cx,cy - separated by commas

        NOTE: top-left is (0,0)
    """    
    with open(intrins_path, 'r') as f:
        intrins_ = f.readlines()[0].split()[0].split(',')
        intrins_ = [float(i) for i in intrins_]
        fx, fy, cx, cy = intrins_

    intrins = torch.tensor([
        [fx, 0,cx],
        [ 0,fy,cy],
        [ 0, 0, 1]
    ], dtype=dtype, device=device)

    return intrins


def get_fov(fx, fy, cx, cy, H, W):
    """ compute horizontal and vertical field-of-view from intrins

        NOTE: top-left is (0,0)
    """
    cx += 0.5
    cy += 0.5
    fov_x = np.rad2deg(np.arctan((W - cx) / fx) + np.arctan((cx) / fx))
    fov_y = np.rad2deg(np.arctan((H - cy) / fy) + np.arctan((cy) / fy))
    return fov_x, fov_y


def get_ray_array(H, W, intrins, flatten=True):
    """ get ray array
        multiplying the output by per-pixel depth would give you the camera-coordinates of each pixel

        NOTE: intrins should be a torch tensor of shape (B, 3, 3)
        NOTE: top-left is (0,0)
    """
    assert torch.is_tensor(intrins) and intrins.ndim == 3
    B, _, _ = intrins.shape

    fx = intrins[:, 0, 0].unsqueeze(-1).unsqueeze(-1)   # (B, 1, 1)
    fy = intrins[:, 1, 1].unsqueeze(-1).unsqueeze(-1)   # (B, 1, 1)
    cx = intrins[:, 0, 2].unsqueeze(-1).unsqueeze(-1)   # (B, 1, 1)
    cy = intrins[:, 1, 2].unsqueeze(-1).unsqueeze(-1)   # (B, 1, 1)

    x_range = torch.cat([torch.arange(W, dtype=intrins.dtype, device=intrins.device).view(1, 1, W)] * H, axis=1).repeat(B,1,1)  # (B, H, W)
    y_range = torch.cat([torch.arange(H, dtype=intrins.dtype, device=intrins.device).view(1, H, 1)] * W, axis=2).repeat(B,1,1)  # (B, H, W)

    # B, 3, H, W
    ray_array = torch.ones((B, 3, H, W), dtype=intrins.dtype, device=intrins.device)
    ray_array[:, 0, :, :] = (x_range - cx) / fx
    ray_array[:, 1, :, :] = (y_range - cy) / fy

    if flatten:
        ray_array = ray_array.view(B, 3, H*W)

    return ray_array


def get_cam_coords(intrins_inv, depth):
    """ camera coordinates from intrins_inv and depth
    
        NOTE: intrins_inv should be a torch tensor of shape (B, 3, 3)
        NOTE: depth should be a torch tensor of shape (B, 1, H, W)
        NOTE: top-left is (0,0)
    """
    assert torch.is_tensor(intrins_inv) and intrins_inv.ndim == 3
    assert torch.is_tensor(depth) and depth.ndim == 4
    assert intrins_inv.dtype == depth.dtype
    assert intrins_inv.device == depth.device
    B, _, H, W = depth.size()

    u_range = torch.arange(W, dtype=depth.dtype, device=depth.device).view(1, W).expand(H, W) # (H, W)
    v_range = torch.arange(H, dtype=depth.dtype, device=depth.device).view(H, 1).expand(H, W) # (H, W)
    ones = torch.ones(H, W, dtype=depth.dtype, device=depth.device)
    pixel_coords = torch.stack((u_range, v_range, ones), dim=0).unsqueeze(0).repeat(B,1,1,1)  # (B, 3, H, W)
    pixel_coords = pixel_coords.view(B, 3, H*W)  # (B, 3, H*W)

    cam_coords = intrins_inv.bmm(pixel_coords).view(B, 3, H, W)
    cam_coords = cam_coords * depth
    return cam_coords


def pix_to_src_coords(src_pix, new_H, new_W, orig_H, orig_W):
    """ src_pix: homogeneous pixel coordinates
        src_coords: used for F.grid_sample (align_corners=False)
    """
    src_pix = src_pix[:2, :] / src_pix[2:, :]

    src_coords = torch.FloatTensor(1, new_H, new_W, 2).fill_(0)
    src_coords[0, :, :, 0] = src_pix[0, :].reshape(new_H, new_W) + 0.5
    src_coords[0, :, :, 1] = src_pix[1, :].reshape(new_H, new_W) + 0.5
    v_center = orig_H / 2.
    u_center = orig_W / 2.
    src_coords[:, :, :, 0] = (src_coords[:, :, :, 0] - u_center) / u_center
    src_coords[:, :, :, 1] = (src_coords[:, :, :, 1] - v_center) / v_center

    src_coords[src_coords > 2.0] = 2.0
    src_coords[src_coords < -2.0] = -2.0
    src_coords[torch.isinf(src_coords)] = 2.0
    src_coords[torch.isnan(src_coords)] = 2.0
    return src_coords


def zbuffer_to_radial(zbuffer, intrins, H, W):
    """ convert zbuffer to radial 
        radial: Euclidean distance from the camera center

        NOTE: zbuffer should be a torch tensor of shape (B, 1, H, W)
        NOTE: intrins should be a torch tensor of shape (B, 3, 3)
        NOTE: top-left is (0,0)
    """
    assert torch.is_tensor(zbuffer) and zbuffer.ndim == 4 
    assert torch.is_tensor(intrins) and intrins.ndim == 3

    ray_array = get_ray_array(H, W, intrins, flatten=False)    # (B, 3, H, W)
    cam_coord = ray_array * zbuffer
    radial = torch.linalg.norm(cam_coord, dim=1, keepdim=True)
    return radial


def radial_to_zbuffer(radial, intrins, H, W):
    """ convert radial to zbuffer
        radial: Euclidean distance from the camera center

        NOTE: radial should be a torch tensor of shape (B, 1, H, W)
        NOTE: intrins should be a torch tensor of shape (B, 3, 3)
        NOTE: top-left is (0,0)
    """
    assert torch.is_tensor(radial) and radial.ndim == 4 
    assert torch.is_tensor(intrins) and intrins.ndim == 3

    ray_array = get_ray_array(H, W, intrins, flatten=False)    # (B, 3, H, W)
    ray_norm = torch.linalg.norm(ray_array, dim=1, keepdim=True)
    zbuffer = radial / ray_norm
    return zbuffer

