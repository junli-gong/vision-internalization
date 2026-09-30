---
dataset_info:
  features:
  - name: image
    dtype:
      array3_d:
        shape:
        - 3
        - 288
        - 384
        dtype: float32
  - name: segmentation
    dtype:
      array2_d:
        shape:
        - 288
        - 384
        dtype: int64
  - name: depth
    dtype:
      array3_d:
        shape:
        - 1
        - 288
        - 384
        dtype: float32
  - name: normal
    dtype:
      array3_d:
        shape:
        - 3
        - 288
        - 384
        dtype: float32
  - name: noise
    dtype:
      array3_d:
        shape:
        - 1
        - 288
        - 384
        dtype: float32
  splits:
  - name: train
    num_bytes: 3525109500
    num_examples: 795
  - name: val
    num_bytes: 2899901400
    num_examples: 654
  download_size: 2971250125
  dataset_size: 6425010900
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
task_categories:
- depth-estimation
- image-segmentation
- image-feature-extraction
size_categories:
- 1K<n<10K
---


This is the NYUv2 dataset for scene understanding tasks.
I downloaded the original data from the [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6d0a89f4ca1347d8af5f/?dl=1) and transformed it into Huggingface Dataset.
Credit to [ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning](http://arxiv.org/abs/2301.12618).

## Dataset Information

This data contains two splits: 'train' and 'val' (used as test dataset).
Each sample in the dataset has 5 items: 'image', 'segmentation', 'depth', 'normal', and 'noise'.
The noise is generated using `torch.rand()`.


## Usage

```python
dataset = load_dataset('tanganke/nyuv2')
dataset = dataset.with_format('torch') # this will convert the items into `torch.Tensor` objects
```

this will return a `DatasetDict`:

```python
DatasetDict({
    train: Dataset({
        features: ['image', 'segmentation', 'depth', 'normal', 'noise'],
        num_rows: 795
    })
    val: Dataset({
        features: ['image', 'segmentation', 'depth', 'normal', 'noise'],
        num_rows: 654
    })
})
```

The features:

```python
{'image': Array3D(shape=(3, 288, 384), dtype='float32', id=None),
 'segmentation': Array2D(shape=(288, 384), dtype='int64', id=None),
 'depth': Array3D(shape=(1, 288, 384), dtype='float32', id=None),
 'normal': Array3D(shape=(3, 288, 384), dtype='float32', id=None),
 'noise': Array3D(shape=(1, 288, 384), dtype='float32', id=None)}
```