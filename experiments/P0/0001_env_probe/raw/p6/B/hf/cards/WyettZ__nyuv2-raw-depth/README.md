---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: depth
    dtype:
      array2_d:
        shape:
        - 448
        - 448
        dtype: float32
  - name: mask
    dtype:
      array2_d:
        shape:
        - 448
        - 448
        dtype: bool
  - name: split
    dtype: string
  - name: index
    dtype: int32
  - name: path
    dtype: string
  - name: orig_h
    dtype: int32
  - name: orig_w
    dtype: int32
  - name: input_h
    dtype: int32
  - name: input_w
    dtype: int32
  splits:
  - name: train
    num_bytes: 868931192
    num_examples: 797
  - name: test
    num_bytes: 710685538
    num_examples: 652
  download_size: 1555497334
  dataset_size: 1579616730
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
