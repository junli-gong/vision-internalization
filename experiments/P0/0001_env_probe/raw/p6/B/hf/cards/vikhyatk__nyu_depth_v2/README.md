---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
dataset_info:
  features:
  - name: image
    dtype: image
  - name: depth_map
    dtype: image
  splits:
  - name: train
    num_bytes: 78426358344.0
    num_examples: 47584
  - name: validation
    num_bytes: 1041123383.0
    num_examples: 654
  download_size: 32624898041
  dataset_size: 79467481727.0
---
# Dataset Card for "nyu_depth_v2"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)