---
dataset_info:
  features:
  - name: image
    dtype:
      image:
        mode: RGB
  - name: segmentations
    sequence:
      image:
        mode: RGB
  - name: instances
    sequence:
      image:
        mode: L
  - name: filename
    dtype: string
  - name: folder
    dtype: string
  - name: source
    struct:
    - name: folder
      dtype: string
    - name: filename
      dtype: string
    - name: origin
      dtype: string
  - name: scene
    sequence: string
  - name: objects
    list:
    - name: id
      dtype: uint16
    - name: name
      dtype: string
    - name: name_ndx
      dtype: uint16
    - name: hypernym
      sequence: string
    - name: raw_name
      dtype: string
    - name: attributes
      dtype: string
    - name: depth_ordering_rank
      dtype: uint16
    - name: occluded
      dtype: bool
    - name: crop
      dtype: bool
    - name: parts
      struct:
      - name: is_part_of
        dtype: uint16
      - name: part_level
        dtype: uint8
      - name: has_parts
        sequence: uint16
    - name: polygon
      struct:
      - name: x
        sequence: uint16
      - name: 'y'
        sequence: uint16
      - name: click_date
        sequence: timestamp[us]
    - name: saved_date
      dtype: timestamp[us]
  splits:
  - name: train
    num_bytes: 4812448179.314
    num_examples: 25574
  - name: validation
    num_bytes: 464280715
    num_examples: 2000
  download_size: 5935251309
  dataset_size: 5276728894.314
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
license: bsd
task_categories:
- image-segmentation
task_ids:
- instance-segmentation
language:
- en
tags:
- MIT
- CSAIL
- panoptic
pretty_name: ADE20K
size_categories:
- 10K<n<100K
paperswithcode_id: ade20k
multilinguality:
- monolingual
annotations_creators:
- crowdsourced
- expert-generated
language_creators:
- found
---

# ADE20K Dataset

[![](https://groups.csail.mit.edu/vision/datasets/ADE20K/assets/images/examples.png)](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

## Dataset Description

- **Homepage:** [MIT CSAIL ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- **Repository:** [github:CSAILVision/ADE20K](https://github.com/CSAILVision/ADE20K)

## Description

ADE20K is composed of more than 27K images from the SUN and Places databases.
Images are fully annotated with objects, spanning over 3K object categories.
Many of the images also contain object parts, and parts of parts.
We also provide the original annotated polygons, as well as object instances for amodal segmentation.
Images are also anonymized, blurring faces and license plates.

## Images

MIT, CSAIL does not own the copyright of the images. If you are a researcher or educator who wish to have a copy of the original images for non-commercial research and/or educational use, we may provide you access by filling a request in our site. You may use the images under the following terms:
  1. Researcher shall use the Database only for non-commercial research and educational purposes. MIT makes no representations or warranties regarding the Database, including but not limited to warranties of non-infringement or fitness for a particular purpose.
  2. Researcher accepts full responsibility for his or her use of the Database and shall defend and indemnify MIT, including their employees, Trustees, officers and agents, against any and all claims arising from Researcher's use of the Database, including but not limited to Researcher's use of any copies of copyrighted images that he or she may create from the Database.
  3. Researcher may provide research associates and colleagues with access to the Database provided that they first agree to be bound by these terms and conditions.
  4. MIT reserves the right to terminate Researcher's access to the Database at any time.
  5. If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.

## Software and Annotations

The MIT CSAIL website, image annotations and the software provided belongs to MIT CSAIL and is licensed under a [Creative Commons BSD-3 License Agreement](https://opensource.org/licenses/BSD-3-Clause). 

Copyright 2019 MIT, CSAIL
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Citations

```bibtex
@inproceedings{8100027,
  title      = {Scene Parsing through ADE20K Dataset},
  author     = {Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
  year       = 2017,
  booktitle  = {2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  volume     = {},
  number     = {},
  pages      = {5122--5130},
  doi        = {10.1109/CVPR.2017.544},
  keywords   = {Image segmentation;Semantics;Sun;Labeling;Visualization;Neural networks;Computer vision}
}
@misc{zhou2018semantic,
  title         = {Semantic Understanding of Scenes through the ADE20K Dataset},
  author        = {Bolei Zhou and Hang Zhao and Xavier Puig and Tete Xiao and Sanja Fidler and Adela Barriuso and Antonio Torralba},
  year          = 2018,
  eprint        = {1608.05442},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CV}
}
```