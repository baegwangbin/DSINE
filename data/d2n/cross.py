import torch
import torch.nn.functional as F


def d2n_tblr(points: torch.Tensor, 
             k: int = 3, 
             d_min: float = 1e-3, 
             d_max: float = 10.0) -> torch.Tensor:
    """ points:     3D points in camera coordinates, shape: (B, 3, H, W)
        k:          neighborhood size
            e.g.)   If k=3, 3x3 neighborhood is used. Two vectors are defined by doing (top-bottom) and (left-right) 
                    Then the normals are computed via cross-product
        d_min/max:  Range of valid depth values 
    """
    k = (k - 1) // 2

    B, _, H, W = points.size()
    points_pad = F.pad(points, (k,k,k,k), mode='constant', value=0)             # (B, 3, k+H+k, k+W+k)
    valid_pad = (points_pad[:,2:,:,:] > d_min) & (points_pad[:,2:,:,:] < d_max) # (B, 1, k+H+k, k+W+k)
    valid_pad = valid_pad.float()

    # vertical vector (top - bottom)
    vec_vert = points_pad[:, :, :H, k:k+W] - points_pad[:, :, 2*k:2*k+H, k:k+W]   # (B, 3, H, W)

    # horizontal vector (left - right)
    vec_hori = points_pad[:, :, k:k+H, :W] - points_pad[:, :, k:k+H, 2*k:2*k+W]   # (B, 3, H, W)

    # valid_mask (all five depth values - center/top/bottom/left/right should be valid)
    valid_mask = valid_pad[:, :, k:k+H,     k:k+W       ] * \
                 valid_pad[:, :, :H,        k:k+W       ] * \
                 valid_pad[:, :, 2*k:2*k+H, k:k+W       ] * \
                 valid_pad[:, :, k:k+H,     :W          ] * \
                 valid_pad[:, :, k:k+H,     2*k:2*k+W   ]
    valid_mask = valid_mask > 0.5
    
    # get cross product (B, 3, H, W)
    cross_product = - torch.linalg.cross(vec_vert, vec_hori, dim=1)
    normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)
   
    return normal, valid_mask
