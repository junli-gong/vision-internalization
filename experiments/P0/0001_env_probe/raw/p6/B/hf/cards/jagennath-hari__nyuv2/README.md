---
language:
- en
pretty_name: NYUv2
tags:
- robotics
license: mit
task_categories:
- depth-estimation
- image-segmentation
- image-feature-extraction
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: id
    dtype: string
  - name: rgb
    dtype: image
  - name: depth
    dtype: image
  - name: semantic
    dtype: image
  - name: instance
    dtype: image
  splits:
  - name: train
    num_bytes: 2237596040.926
    num_examples: 1014
  - name: val
    num_bytes: 639427936.0
    num_examples: 290
  - name: test
    num_bytes: 320361261.0
    num_examples: 145
  download_size: 1284641786
  dataset_size: 3197385237.926
---

# NYUv2

This is an **unofficial and preprocessed version** of [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) made available for easier integration with modern ML workflows. The dataset was converted from the original `.mat` format into a split structure with embedded RGB images, depth maps, semantic masks, and instance masks in Hugging Face-compatible format.

## 📸 Sample Visualization

<div align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <img src="assets/RGB.png" alt="RGB" width="100%"/>
        <div><b>RGB</b></div>
      </td>
      <td align="center" width="33%">
        <img src="assets/Depth.png" alt="Depth" width="100%"/>
        <div><b>Depth (Jet colormap)</b></div>
      </td>
      <td align="center" width="33%">
        <img src="assets/Semantic.png" alt="Semantic" width="100%"/>
        <div><b>Semantic Mask</b></div>
      </td>
    </tr>
  </table>
</div>

## Dataset Description

- **Homepage:** [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
- **Paper:** [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf)

NYUv2 is a benchmark RGB-D dataset widely used for scene understanding tasks such as:

- Indoor semantic segmentation
- Depth estimation
- Instance segmentation

This version has been preprocessed to include aligned:
- Undistorted RGB images (`.png`)
- Depth maps in millimeters (`.tiff`, `uint16`)
- Semantic masks (`.tiff`, scaled `uint16`)
- Instance masks (`.tiff`, scaled `uint16`)

Each sample is annotated with a consistent `id` and split across train/val/test.

## 🧾 Dataset Metadata

Additional files included:
- `camera_params.json` — camera intrinsics and distortion
- `class_names.json` — mapping from class IDs to human-readable names
- `scaling_factors.json` — used for metric depth and label/mask de-scaling during training

## 🚀 How to Use

You can load the dataset using the `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("jagennath-hari/nyuv2", split="train")
sample = dataset[0]

# Access fields
rgb = sample["rgb"]
depth = sample["depth"]
semantic = sample["semantic"]
instance = sample["instance"]
```

### 🔄 Recover Original Values from TIFF Images

The dataset uses .tiff format for all dense outputs to preserve precision and visual compatibility. Here’s how to revert them back to their original values:

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np
import json
import os

# Load sample
dataset = load_dataset("jagennath-hari/nyuv2", split="train")
sample = dataset[0]

# Download and load scaling metadata
local_dir = snapshot_download(
    repo_id="jagennath-hari/nyuv2",
    repo_type="dataset",
    allow_patterns="scaling_factors.json"
)
with open(os.path.join(local_dir, "scaling_factors.json")) as f:
    scale = json.load(f)

depth_scale = scale["depth_scale"]
label_max = scale["label_max_value"]
instance_max = scale["instance_max_value"]

# === Unscale depth (mm → m)
depth_img = np.array(sample["depth"])
depth_m = depth_img.astype(np.float32) / depth_scale

# === Unscale semantic mask
sem_scaled = np.array(sample["semantic"])
semantic_labels = np.round(
    sem_scaled.astype(np.float32) * (label_max / 65535.0)
).astype(np.uint16)

# === Unscale instance mask
inst_scaled = np.array(sample["instance"])
instance_ids = np.round(
    inst_scaled.astype(np.float32) * (instance_max / 65535.0)
).astype(np.uint16)
```

### 📐 Scaling Factors Summary

| Field     | Stored As        | Original Format       | Scaling Method                  | Undo Formula                                       |
|-----------|------------------|------------------------|----------------------------------|----------------------------------------------------|
| `depth`   | `uint16`, mm     | `float32`, meters      | multiplied by `depth_scale`     | `depth / depth_scale`                              |
| `semantic`| `uint16`, scaled | `uint16` class IDs     | scaled by `65535 / label_max`   | `round(mask * (label_max / 65535.0))`              |
| `instance`| `uint16`, scaled | `uint16` instance IDs  | scaled by `65535 / instance_max`| `round(mask * (instance_max / 65535.0))`           |


## 📄 Citation

If you use this dataset, please cite the original authors:

```bibtex
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {ECCV},
  year      = {2012}
}
```