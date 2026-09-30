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
    num_bytes: 81385755
    num_examples: 42404
  - name: testB
    num_bytes: 3284397
    num_examples: 1810
  - name: test
    num_bytes: 3943834
    num_examples: 1975
  - name: validation
    num_bytes: 7355626
    num_examples: 3811
  download_size: 38895129
  dataset_size: 95969612
---
# Dataset Card for "refcoco"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)