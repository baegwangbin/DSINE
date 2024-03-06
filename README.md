
# Rethinking Inductive Biases for Surface Normal Estimation

<p align="center">
  <img width=20% src="https://github.com/baegwangbin/DSINE/raw/main/docs/img/dsine/logo_with_outline.png">
</p>

Official implementation of the paper

> **Rethinking Inductive Biases for Surface Normal Estimation**
>
> CVPR 2024
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