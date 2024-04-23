# Depth-to-Normal

We generated our meta-dataset by collecting RGB-D datasets and converting the depth maps into surface normal maps.

We took [PlaneSVD from Klasing et al.](https://ieeexplore.ieee.org/document/5152493) and added a few modifications to handle depth discontinuties. We encourage you to try using other algorithms as it can potentially improve the quality of the ground truth and hence the performance of the model.

Please see `notes/depth_to_normal.ipynb` for example usage.
