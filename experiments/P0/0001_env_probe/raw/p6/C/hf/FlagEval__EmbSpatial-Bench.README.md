---
dataset_info:
  features:
  - name: data_source
    dtype: string
  - name: scene_id
    dtype: string
  - name: question_id
    dtype: string
  - name: question
    dtype: string
  - name: relation
    dtype: string
  - name: image
    dtype: image
  - name: answer_options
    sequence: string
  - name: answer
    dtype: int32
  - name: objects
    sequence:
    - name: bbox
      sequence: int32
      length: 4
    - name: name
      dtype: string
  - name: image_base64
    dtype: string
  splits:
  - name: test
    num_bytes: 441649188.0
    num_examples: 3640
  download_size: 408685444
  dataset_size: 441649188.0
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

# Introduction

**Disclaimer:** This dataset is organized and adapted from [Phineas476/EmbSpatial-Bench](https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench). The original data was image format and has been converted here into a more accessible and easy-to-use format.

EmbSpatial-Bench is a benchmark for evaluating embodied spatial understanding of LVLMs. The benchmark is automatically derived from embodied scenes and covers 6 spatial relationships from an egocentric perspective. The constructed benchmark comprises a total of 3,640 QA pairs, covering 294 object categories and 6 spatial relationships.


# Data Fields

| Field Name       | Type              | Description                     |
|------------------|-------------------|---------------------------------|
| data_source      | string            | Name or identifier of the data source |
| scene_id         | string            | Unique scene ID                 |
| question_id      | string            | Unique question ID              |
| question         | string            | Text of the question            |
| relation         | string            | Relationship described in the question |
| image            | Image             | Scene image data                |
| answer_options   | sequence<string>  | List of answer options          |
| answer           | int32             | Index of the correct answer     |
| objects          | sequence          | List of objects in the image    |
| bbox             | sequence<int32>   | Bounding box coordinates (x_min, y_min, x_max, y_max) |
| name             | string            | Name of the object              |
| image_base64     | string            | Image data in base64 encoding   |
