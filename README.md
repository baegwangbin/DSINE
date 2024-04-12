
# Rethinking Inductive Biases for Surface Normal Estimation

<p align="center">
  <img width=20% src="https://github.com/baegwangbin/DSINE/raw/main/docs/img/dsine/logo_with_outline.png">
</p>

Official implementation of the paper

> **Rethinking Inductive Biases for Surface Normal Estimation**
>
> CVPR 2024 [oral]
>
> <a href="https://baegwangbin.com" target="_blank">Gwangbin Bae</a> and <a href="https://www.doc.ic.ac.uk/~ajd/" target="_blank">Andrew J. Davison</a>
>
> <a href="https://github.com/baegwangbin/DSINE/raw/main/paper.pdf" target="_blank">[paper.pdf]</a>
<a href="https://arxiv.org/abs/2403.00712" target="_blank">[arXiv]</a> 
<a href="https://www.youtube.com/watch?v=2y9-35c719Y&t=5s" target="_blank">[youtube]</a> 
<a href="https://baegwangbin.github.io/DSINE/" target="_blank">[project page]</a>

## Abstract

Despite the growing demand for accurate surface normal estimation models, existing methods use general-purpose dense prediction models, adopting the same inductive biases as other tasks. In this paper, we discuss the **inductive biases** needed for surface normal estimation and propose to **(1) utilize the per-pixel ray direction** and **(2) encode the relationship between neighboring surface normals by learning their relative rotation**. The proposed method can generate **crisp — yet, piecewise smooth — predictions** for challenging in-the-wild images of arbitrary resolution and aspect ratio. Compared to a recent ViT-based state-of-the-art model, our method shows a stronger generalization ability, despite being trained on an orders of magnitude smaller dataset.

<p align="center">
  <img width=100% src="https://github.com/baegwangbin/DSINE/raw/main/docs/img/fig_comparison.png">
</p>

## Getting Started

Start by installing the dependencies.

```
conda create --name DSINE python=3.10
conda activate DSINE

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install geffnet
python -m pip install glob2
```

Then, download the model weights from <a href="https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm?usp=drive_link" target="_blank">this link</a> and save it under `./checkpoints/`.

## Test on images

* Run `python test.py` to generate predictions for the images under `./samples/img/`. The result will be saved under `./samples/output/`.
* Our model assumes known camera intrinsics, but providing approximate intrinsics still gives good results. For some images in `./samples/img/`, the corresponding camera intrinsics (fx, fy, cx, cy - assuming perspective camera with no distortion) is provided as a `.txt` file. If such a file does not exist, the intrinsics will be approximated, by assuming $60^\circ$ field-of-view.

## Additional instructions

If you want to make contributions to this repo, please make a pull request and add instructions in the following format.

<details>
<summary><b>Using torch hub to predict normal</b> (contribution by <a href="https://github.com/hugoycj" target="_blank">hugoycj</a>)</summary>

```
import torch
import cv2
import numpy as np

# Load the normal predictor model from torch hub
normal_predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)

# Load the input image using OpenCV
image = cv2.imread(args.input, cv2.IMREAD_COLOR)
h, w = image.shape[:2]

# Use the model to infer the normal map from the input image
with torch.inference_mode():
    normal = normal_predictor.infer_cv2(image)[0]  # Output shape: (H, W, 3)
    normal = (normal + 1) / 2  # Convert values to the range [0, 1]

# Convert the normal map to a displayable format
normal = (normal * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)

# Save the output normal map to a file
cv2.imwrite(args.output, normal)
```

If the network is unavailable to retrieve weights, you can use local weights for torch hub as shown below:

```
normal_predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", local_file_path='./checkpoints/dsine.pt', trust_repo=True)
```
</details>


<details>
<summary><b>Generating ground truth surface normals</b></summary>
We provide the code used to generate the ground truth surface normals from ground truth depth maps. See <code>data/d2n/README.md</code> for more detail.
</details>


<details>
<summary><b>About the coordinate system</b></summary>
We use the right-handed coordinate system with (X, Y, Z) = (right, down, front). An important thing to note is that both the ground truth normals and our prediction are the <b>outward normals</b>. For example, in the case of a fronto-parallel wall facing the camera, the normals would be (0, 0, 1), not (0, 0, -1). If you instead need to use the <b>inward normals</b>, please do <code>normals = -normals</code>.
</details>


 

## Citation

If you find our work useful in your research please consider citing our paper:

```
@inproceedings{bae2024dsine,
    title={Rethinking Inductive Biases for Surface Normal Estimation},
    author={Gwangbin Bae and Andrew J. Davison},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```