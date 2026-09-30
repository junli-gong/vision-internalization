---
size_categories:
- 1K<n<10K
task_categories:
- depth-estimation
- image-segmentation
paperswithcode_id: nyuv2
tags:
- depth-estimation
- semantic-segmentation
dataset_info:
  features:
  - name: image
    dtype: image
  - name: depth
    dtype:
      array2_d:
        shape:
        - 640
        - 480
        dtype: float32
  - name: label
    dtype:
      array2_d:
        shape:
        - 640
        - 480
        dtype: int32
  - name: scene
    dtype: string
  - name: scene_type
    dtype: string
  - name: accelData
    sequence: float32
    length: 4
  splits:
  - name: train
    num_bytes: 4096489803
    num_examples: 1449
  download_size: 2972037809
  dataset_size: 4096489803
---

> [!WARNING]
> This repo relies on a data loading script, which is no longer supported with `datasets>=4.0`

# NYU Depth Dataset V2

This is an unofficial Hugging Face downloading script of the [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html). It downloads the data from the original source and converts it to the Hugging Face format.

This dataset contains the 1449 densely labeled pairs of aligned RGB and depth images.


## Dataset Description

- **Homepage:** [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
- **Paper:** [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf)


## Official Description

The NYU-Depth V2 data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect. It features:

* 1449 densely labeled pairs of aligned RGB and depth images
* 464 new scenes taken from 3 cities
* 407,024 new unlabeled frames
* Each object is labeled with a class and an instance number (cup1, cup2, cup3, etc)

This dataset is useful for various computer vision tasks, including depth estimation, semantic segmentation, and instance segmentation.


## Usage

```python
from datasets import load_dataset

dataset = load_dataset("0jl/NYUv2", trust_remote_code=True, split="train")
```


### Common Errors

* `fsspec.exceptions.FSTimeoutError`

  Can occur for `datasets==3.0` when the download takes more than 5 minutes. This increases the timeout to 1 hour:

  ```python
  import datasets, aiohttp
  dataset = datasets.load_dataset(
      "0jl/NYUv2",
      trust_remote_code=True,
      split="train",
      storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
  )
  ```


## Dataset Structure

The dataset contains only one training split with the following features:

- `image`: RGB image (PIL.Image.Image, shape: (640, 480, 3))
- `depth`: Depth map (2D array, shape: (640, 480), dtype: float32)
- `label`: Semantic segmentation labels (2D array, shape: (640, 480), dtype: int32)
- `scene`: Scene name (string)
- `scene_type`: Scene type (string)
- `accelData`: Acceleration data (list, shape: (4,), dtype: float32)


## Citation Information

If you use this dataset, please cite the original paper:

```bibtex
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {Proceedings of the European Conference on Computer Vision},
  year      = {2012}
}
```
