import torch
import torch.nn as nn
import torch.nn.functional as F


class Depth2normal(nn.Module):
    def __init__(self, 
                 d_min: float = 0.0, 
                 d_max: float = 10.0, 
                 k: int = 3, 
                 d: int = 1, 
                 min_nghbr: int = 4, 
                 gamma: float = None, 
                 gamma_exception: bool = False):
        super(Depth2normal, self).__init__()

        # range of valid depth values
        # if the depth is outside this range, it will be considered invalid
        self.d_min = d_min
        self.d_max = d_max

        # neighborhood size, k x k neighborhood around each pixel will be considered
        self.k = k

        # spacing between the nghbrs (dilation)
        self.d = d

        # if the difference between the center depth and nghbr depth is larger than this, it will be ignored
        # e.g. gamma=0.05 means that a nghbr pixel is ignored if its depth is more than 5% different from the center pixel
        self.gamma = gamma  

        # minimum number of nghbr pixels
        # if the number of valid nghbr pixels is below this value, the normals would be considered invalid
        self.min_nghbr = min_nghbr

        # if the normal of a flat surface is near-vertical to the viewing direction, the depth gradient will be very high,
        # and most nghbr pixels would not pass the above "gamma" test
        # this can be a problem when using datasets like Virtual KITTI (i.e. the ones with large horizontal surfaces)
        # if gamma_exception is set to True, 
        # the "gamma" test will be ignored when the number of valid nghbr pixels < self.min_nghbr
        self.gamma_exception = gamma_exception

        # padding for depth map
        self.pad = (k + (k - 1) * (d - 1)) // 2

        # index of the center pixel
        self.center_idx = (k*k - 1) // 2

        # torch Unfold to unfold the depth map
        self.unfold = torch.nn.Unfold(kernel_size=(k, k), padding=self.pad, dilation=d)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """ points: 3D points in camera coordinates, shape: (B, 3, H, W)
        """
        b, _, h, w = points.shape

        # matrix_a (b, h, w, k*k, 3)
        torch_patches = self.unfold(points)                                     # (b, 3*k*k, h, w)
        matrix_a = torch_patches.view(b, 3, self.k * self.k, h, w)              # (b, 3, k*k, h, w)
        matrix_a = matrix_a.permute(0, 3, 4, 2, 1)                              # (b, h, w, k*k, 3)

        # filter by depth
        valid_condition = torch.logical_and(points[:,2:,:,:] > self.d_min, points[:,2:,:,:] < self.d_max)
        valid_condition = valid_condition.float()                               # (B, 1, H, W)
        valid_condition = self.unfold(valid_condition)                          # (b, 1*k*k, h, w)
        valid_condition = valid_condition.view(b, 1, self.k * self.k, h, w)     # (b, 1, k*k, h, w)
        valid_condition = valid_condition.permute(0, 3, 4, 2, 1)                # (b, h, w, k*k, 1)

        # valid_condition (b, h, w, k*k, 1)
        if self.gamma is not None:
            valid_depth_diff = torch.abs(matrix_a[:, :, :, :, 2:] - matrix_a[:, :, :, self.center_idx:self.center_idx+1, 2:]) \
                            / matrix_a[:, :, :, self.center_idx:self.center_idx+1, 2:]
            valid_depth_diff = (valid_depth_diff < self.gamma).float()              # (b, h, w, k*k, 1)

            if self.gamma_exception:
                valid_depth_diff_sum = torch.sum(valid_depth_diff, dim=3, keepdim=True)     # (b, h, w, 1, 1)
                valid_depth_diff_sum = (valid_depth_diff_sum < self.min_nghbr).float()     # (b, h, w, 1, 1)    
                valid_depth_diff = valid_depth_diff + valid_depth_diff_sum
                valid_depth_diff = (valid_depth_diff > 0.5).float()

            valid_condition = valid_condition * valid_depth_diff

        # matrix A (b, h, w, k*k, 4)
        matrix_1 = torch.ones_like(matrix_a[:,:,:,:,0:1])
        matrix_A = torch.cat([matrix_a, matrix_1], dim=-1)

        # fill zero for invalid pixels
        matrix_A_zero = torch.zeros_like(matrix_A)
        matrix_A = torch.where(valid_condition.repeat([1, 1, 1, 1, 4]) > 0.5, matrix_A, matrix_A_zero)

        # transpose
        matrix_At = torch.transpose(matrix_A, 3, 4)

        matrix_A = matrix_A.view(-1, self.k * self.k, 4)    # (b*h*w, k*k, 4)
        matrix_At = matrix_At.view(-1, 4, self.k * self.k)  # (b*h*w, 4, k*k)
        At_A = torch.bmm(matrix_At, matrix_A)               # (b*h*w, 4, 4)

        # eig_val: (b*h*w, 4) / eig_vec: (b*h*w, 4, 4)
        eig_val, eig_vec = torch.linalg.eig(At_A)

        # valid_mask (b*h*w)
        valid_eig = torch.logical_and(torch.sum(eig_val.imag, dim=1) == 0,
                        torch.sum(eig_vec.imag, dim=(1, 2)) == 0)

        # find the smallest eigenvalue
        eig_val = eig_val.real
        eig_vec = eig_vec.real

        idx = torch.argmin(eig_val, dim=1, keepdim=True)  # (b*h*w, 1)
        idx_onehot = torch.zeros_like(eig_val).scatter_(1, idx, 1.) # (b*h*w, 4)
        idx_onehot = idx_onehot.unsqueeze(1).repeat(1, 4, 1)

        # normal (b, 3, h, w)
        normal = torch.sum(eig_vec * idx_onehot, dim=2)
        normal = normal.view(b, h, w, 4).permute(0, 3, 1, 2).contiguous()
        normal = F.normalize(normal[:,:3,:,:], p=2.0, dim=1, eps=1e-12)

        # flip if needed
        flip = torch.sign(torch.sum(normal * points, dim=1, keepdim=True))
        normal = normal * flip

        # valid_mask1 (b, 1, h, w): center pixel valid depth
        valid_mask1 = valid_condition[:,:,:,self.center_idx,0].unsqueeze(1)

        # valid_mask2 (b, 1, h, w): sufficient number of valid neighbors
        valid_mask2 = torch.sum(valid_condition[..., 0], dim=3).unsqueeze(1) >= self.min_nghbr

        # valid_mask3 (b, 1, h, w): eigenvalue real
        valid_mask3 = valid_eig.view(b, h, w).unsqueeze(1)

        # valid_mask4 (b, 1, h, w):
        valid_mask4 = torch.norm(normal, p=2, dim=1, keepdim=True) > 0.5

        # valid_mask
        valid_mask = valid_mask1 * valid_mask2 * valid_mask3 * valid_mask4
        
        return normal, valid_mask > 0.5


