---
dataset_info:
  features:
  - name: sent_ids
    sequence: int64
  - name: file_name
    dtype: string
  - name: ann_id
    dtype: int64
  - name: ref_id
    dtype: int64
  - name: image_id
    dtype: int64
  - name: split
    dtype: string
  - name: sentences
    list:
    - name: raw
      dtype: string
    - name: sent
      dtype: string
    - name: sent_id
      dtype: int64
    - name: tokens
      sequence: string
  - name: category_id
    dtype: int64
  - name: raw_anns
    dtype: string
  - name: raw_image_info
    dtype: string
  - name: raw_sentences
    dtype: string
  - name: image_path
    dtype: string
  - name: bbox
    sequence: float64
  - name: captions
    sequence: string
  - name: global_image_id
    dtype: string
  - name: anns_id
    dtype: string
  splits:
  - name: train
    num_bytes: 81937869
    num_examples: 42278
  - name: testB
    num_bytes: 3273927
    num_examples: 1798
  - name: test
    num_bytes: 3969265
    num_examples: 1975
  - name: validation
    num_bytes: 7399541
    num_examples: 3805
  download_size: 39772801
  dataset_size: 96580602
---
# Dataset Card for "refcocoplus"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)