---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: dataset_name
    dtype: string
  - name: categories
    list: string
  - name: bboxes
    list:
      list: float32
      length: 4
  - name: labels
    list: string
  splits:
  - name: test
    num_bytes: 320261336
    num_examples: 4938
  download_size: 214297376
  dataset_size: 320261336
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
