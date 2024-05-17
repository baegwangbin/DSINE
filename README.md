
# Rethinking Inductive Biases for Surface Normal Estimation

<p align="center">
  <img width=20% src="https://github.com/baegwangbin/DSINE/raw/main/docs/img/dsine/logo_with_outline.png">
</p>

Official implementation of the paper

> **Rethinking Inductive Biases for Surface Normal Estimation** \
> CVPR 2024 [oral] \
> <a href="https://baegwangbin.com" target="_blank">Gwangbin Bae</a> and <a href="https://www.doc.ic.ac.uk/~ajd/" target="_blank">Andrew J. Davison</a> \
> <a href="https://github.com/baegwangbin/DSINE/raw/main/paper.pdf" target="_blank">[paper.pdf]</a>
<a href="https://arxiv.org/abs/2403.00712" target="_blank">[arXiv]</a> 
<a href="https://www.youtube.com/watch?v=2y9-35c719Y&t=5s" target="_blank">[youtube]</a> 
<a href="https://baegwangbin.github.io/DSINE/" target="_blank">[project page]</a>

## Abstract

Despite the growing demand for accurate surface normal estimation models, existing methods use general-purpose dense prediction models, adopting the same inductive biases as other tasks. In this paper, we discuss the **inductive biases** needed for surface normal estimation and propose to **(1) utilize the per-pixel ray direction** and **(2) encode the relationship between neighboring surface normals by learning their relative rotation**. The proposed method can generate **crisp — yet, piecewise smooth — predictions** for challenging in-the-wild images of arbitrary resolution and aspect ratio. Compared to a recent ViT-based state-of-the-art model, our method shows a stronger generalization ability, despite being trained on an orders of magnitude smaller dataset.

<p align="center">
  <img width=100% src="https://github.com/baegwangbin/DSINE/raw/main/docs/img/fig_comparison_new.png">
</p>

## Getting started

We provide the instructions in **four steps** (click "▸" to expand). For example, if you just want to test DSINE on some images, you can stop after **Step 1**. This would minimize the amount of installation/downloading. 

<details>
<summary><b>Step 1. Test DSINE on some images</b> (requires minimal dependencies)</summary>

Start by installing dependencies.

```
conda create --name DSINE python=3.10
conda activate DSINE

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
python -m pip install geffnet
```

Then, download the model weights from <a href="https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm?usp=drive_link" target="_blank">this link</a> and save it under `projects/dsine/checkpoints/`. Note that it should maintain the same folder structure as the google drive. For example, `checkpoints/exp001_cvpr2024/dsine.pt` (in google drive) is our best model. It should be saved as `projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt`. The corresponding config file is `projects/dsine/experiments/exp001_cvpr2024/dsine.txt`.

The models under `checkpoints/exp002_kappa/` (in google drive) are the ones that can also estimate uncertainty. 

Then, move to the folder `projects/dsine/`, and run

```
python test_minimal.py ./experiments/exp001_cvpr2024/dsine.txt
```

This will generate predictions for the images under `projects/dsine/samples/img/`. The result will be saved under `projects/dsine/samples/output/`.

Our model assumes known camera intrinsics, but providing approximate intrinsics still gives good results. For some images in `projects/dsine/samples/img/`, the corresponding camera intrinsics (fx, fy, cx, cy - assuming perspective camera with no distortion) is provided as a `.txt` file. If such a file does not exist, the intrinsics will be approximated, by assuming $60^\circ$ field-of-view.
</details>


<details>
<summary><b>Step 2. Test DSINE on benchmark datasets & run a real-time demo</b></summary>

Install additional dependencies.

```
python -m pip install tensorboard
python -m pip install opencv-python
python -m pip install matplotlib

python -m pip install pyrealsense2    # needed only for demo using a realsense camera
python -m pip install vidgear         # needed only for demo on YouTube videos
python -m pip install yt_dlp          # needed only for demo on YouTube videos
python -m pip install mss             # needed only for demo on screen capture
```

Download the evaluation datasets (`dsine_eval.zip`) from <a href="https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm?usp=drive_link" target="_blank">this link</a>. 

**NOTE:** By downloading the dataset, you are agreeing to the respective LICENSE of each dataset. The link to the dataset can be found in the respective `readme.txt`.

If you go to `projects/__init__.py`, there is a variable called `DATASET_DIR` and `EXPERIMENT_DIR`:

* `DATASET_DIR` is where your dataset should be stored. For example, the `dsine_eval` dataset (downloaded from the link above) should be saved under `DATASET_DIR/dsine_eval`. Update this variable.
* `EXPERIMENT_DIR` is where the experiments (e.g. model weights, log, etc) will be saved. Update this variable.

Then, move to the folder `projects/dsine/`, and run:

```python
# getting benchmark performance on the six evaluation datasets
python test.py ./experiments/exp001_cvpr2024/dsine.txt --mode benchmark

# getting benchmark performance on the six evaluation datasets (with visualization)
# it will be saved under EXPERIMENT_DIR/dsine/exp001_cvpr2024/dsine/test/
python test.py ./experiments/exp001_cvpr2024/dsine.txt --mode benchmark --visualize

# generate predictions for the images in `projects/dsine/samples/img/
python test.py ./experiments/exp001_cvpr2024/dsine.txt --mode samples

# measure the throughput (inference speed) on your device
python test.py ./experiments/exp001_cvpr2024/dsine.txt --mode throughput
```

You can also run a real-time demo by running:

```python
# captures your screen and makes prediction
python test.py ./experiments/exp001_cvpr2024/dsine.txt --mode screen

# demo using webcam
python test.py ./experiments/exp001_cvpr2024/dsine.txt --mode webcam

# demo using a realsense camera
python test.py ./experiments/exp001_cvpr2024/dsine.txt --mode rs

# demo on a Youtube video (replace with a different link)
python test.py ./experiments/exp001_cvpr2024/dsine.txt --mode https://www.youtube.com/watch?v=X-iEq8hWd6k
```

For each input option, there are some additional parameters. See `projects/dsine/test.py` for more information.

You can also try building your own real-time demo. Please see [this notebook](https://github.com/baegwangbin/DSINE/blob/main/notes/real_time_demo.ipynb) for more information.
</details>


<details>
<summary><b>Step 3. Train DSINE</b></summary>

In `projects/dsine/`, run:

```python
python train.py ./experiments/exp000_test/test.txt
```

And do `tensorboard --logdir EXPERIMENT_DIR/dsine/exp000_test/test/log` to open the tensorboard.

This will train the model on the train split of the NYUv2 dataset, which should be under `DATASET_DIR/dsine_eval/nyuv2/train/`. There are only 795 images here, and the performance will not be good. To get better results you need to:

**(1) Create a custom dataloader**

>We are checking if we can release the entire training dataset (~400GB). Before the release, you can try building your custom dataloader. You need to define a `get_sample(args, sample_path, info)` function and provide a data split in `data/datasets`. Check how they are defined/provided for other datasets. You also need to update `projects/baseline_normal/dataloader.py` so the newly defined `get_sample` function can be used.

**(2) Generate GT surface normals** (optional)

>In case your dataset does not come with ground truth surface normal maps, you can try generating them from the ground truth depth maps. Please see [this notebook](https://github.com/baegwangbin/DSINE/blob/main/notes/depth_to_normal.ipynb) for more information.

**(3) Customize data augmentation**

>In case you are using synthetic images, you need the right set of data augmentation functions to minimize the synthetic-to-real domain gap. We provide a wide range of augmentation functions, but the hyperparameters are not finetuned and you can potentially get better results by finetuning them. Please see [this notebook](https://github.com/baegwangbin/DSINE/blob/main/notes/visualize_augmentation.ipynb) for more information.

</details>


<details>
<summary><b>Step 4. Start your own surface normal estimation project</b></summary>

If you want to start your own surface normal estimation project, you can do so very easily. 

First of all, have a look at `projects/baseline_normal`. This is a place where you can try different CNN architectures without worrying about the camera intrinsics and rotation estimation. You can try popular architectures like U-Net, and try different backbones. In this folder, you can run: 

```python
python train.py ./experiments/exp000_test/test.txt
```

The project-specific `config` is defined in `projects/baseline_normal/config.py`. Default config, which is shared across all projects are in `projects/__init__.py`.

The dataloaders are in `projects/baseline_normal/dataloader.py`. We use the same dataloaders in `dsine` project, so we don't have `projects/dsine/dataloader.py`.

The losses are defined in `projects/baseline_normal/losses.py`. These are building blocks for your custom loss functions in your own project. For example, in the DSINE project, we produce a list of predictions and the loss is the weighted sum of the losses computed for each prediction. You can see how this is done in `projects/dsine/losses.py`.

You can start a new project by copying the folder `projects/dsine` to create `projects/NEW_PROJECT_NAME`. Then, update the `config.py` and `losses.py`.

Lastly, you can should `train.py` and `test.py`. For things that should be different in different projects, we made a note like following:

```python
#↓↓↓↓
#NOTE: forward pass
img = data_dict['img'].to(device)
intrins = data_dict['intrins'].to(device)
...
pred_list = model(img, intrins=intrins, mode='test')
norm_out = pred_list[-1]
#↑↑↑↑
```

Search for the arrows (↓↓↓↓/↑↑↑↑) to see where things should be modified in different projects.

The test commands above (e.g. for getting the benchmark performance & running real-time demo) should apply the same for all projects.
</details>

## Additional instructions

If you want to make contributions to this repo, please make a pull request and add instructions in the following format.

<details>
<summary><b>Using torch hub to predict normal</b> (contribution by <a href="https://github.com/hugoycj" target="_blank">hugoycj</a>)</summary>

NOTE: the code below is deprecated and should be modified (as the folder structure has changed).

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
We provide the code used to generate the ground truth surface normals from ground truth depth maps. See <a href="https://github.com/baegwangbin/DSINE/blob/main/notes/depth_to_normal.ipynb" target="_blank">this notebook</a> for more information.
</details>


<details>
<summary><b>About the coordinate system</b></summary>
We use the right-handed coordinate system with (X, Y, Z) = (right, down, front). An important thing to note is that both the ground truth normals and our prediction are the <b>outward normals</b>. For example, in the case of a fronto-parallel wall facing the camera, the normals would be (0, 0, 1), not (0, 0, -1). If you instead need to use the <b>inward normals</b>, please do <code>normals = -normals</code>.
</details>


<details>
<summary><b>Sharing your model weights</b></summary>
If you wish to share your model weights, please make a pull request by providing the corresponding config file and the link to the weights.
</details>

## Citation

If you find our work useful in your research please consider citing our paper:

```
@inproceedings{bae2024dsine,
    title     = {Rethinking Inductive Biases for Surface Normal Estimation},
    author    = {Gwangbin Bae and Andrew J. Davison},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}
```

If you use the models that also estimate the uncertainty, please also cite the following paper, where we introduced the loss function:

```
@InProceedings{bae2021eesnu,
    title     = {Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation}
    author    = {Gwangbin Bae and Ignas Budvytis and Roberto Cipolla},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year      = {2021}                         
}
```
