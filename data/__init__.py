""" data sample
"""
class Sample():
    def __init__(self, img=None, 
                 depth=None, depth_mask=None,
                 normal=None, normal_mask=None, 
                 intrins=None, flipped=False,
                 dataset_name='dataset', scene_name='scene', img_name='img', 
                 info={}):
 
        self.img = img                  # input image

        self.depth = depth              # depth - GT
        self.depth_mask = depth_mask    # depth - valid_mask

        self.normal = normal            # surface normals - GT
        self.normal_mask = normal_mask  # surface normals - valid_mask
        
        self.intrins = intrins          # camera intrinsics
        self.flipped = flipped          # True when the image is flipped during augmentation

        self.dataset_name = dataset_name
        self.scene_name = scene_name
        self.img_name = img_name

        # other info (this is a dict containing any additional information)
        self.info = info