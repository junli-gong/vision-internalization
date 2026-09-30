---
dataset_info:
  features:
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
  - name: file_name
    dtype: string
  - name: category_id
    dtype: int64
  - name: ann_id
    dtype: int64
  - name: sent_ids
    sequence: int64
  - name: ref_id
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
  - name: test
    num_bytes: 10341980
    num_examples: 5023
  - name: train
    num_bytes: 87352599
    num_examples: 42226
  - name: validation
    num_bytes: 5236723
    num_examples: 2573
  download_size: 45968855
  dataset_size: 102931302
---
# Dataset Card for "refcocog"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)