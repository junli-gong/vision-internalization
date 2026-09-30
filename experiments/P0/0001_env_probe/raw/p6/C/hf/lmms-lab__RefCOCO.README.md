---
dataset_info:
  features:
  - name: question_id
    dtype: string
  - name: image
    dtype: image
  - name: question
    dtype: string
  - name: answer
    sequence: string
  - name: segmentation
    sequence: float32
  - name: bbox
    sequence: float32
  - name: iscrowd
    dtype: int8
  - name: file_name
    dtype: string
  splits:
  - name: val
    num_bytes: 1548717880.0
    num_examples: 8811
  - name: test
    num_bytes: 876787122.0
    num_examples: 5000
  - name: testA
    num_bytes: 340830323.0
    num_examples: 1975
  - name: testB
    num_bytes: 317959580.0
    num_examples: 1810
  download_size: 2278337287
  dataset_size: 3084294905.0
configs:
- config_name: default
  data_files:
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
  - split: testA
    path: data/testA-*
  - split: testB
    path: data/testB-*
---


<p align="center" width="100%">
<img src="https://i.postimg.cc/g0QRgMVv/WX20240228-113337-2x.png"  width="100%" height="80%">
</p>

# Large-scale Multi-modality Models Evaluation Suite

> Accelerating the development of large-scale multi-modality models (LMMs) with `lmms-eval`

🏠 [Homepage](https://lmms-lab.github.io/) | 📚 [Documentation](docs/README.md) | 🤗 [Huggingface Datasets](https://huggingface.co/lmms-lab)

# This Dataset

This is a formatted version of [RefCOCO](https://github.com/lichengunc/refer). It is used in our `lmms-eval` pipeline to allow for one-click evaluations of large multi-modality models.

```
@inproceedings{kazemzadeh-etal-2014-referitgame,
    title = "{R}efer{I}t{G}ame: Referring to Objects in Photographs of Natural Scenes",
    author = "Kazemzadeh, Sahar  and
      Ordonez, Vicente  and
      Matten, Mark  and
      Berg, Tamara",
    editor = "Moschitti, Alessandro  and
      Pang, Bo  and
      Daelemans, Walter",
    booktitle = "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})",
    month = oct,
    year = "2014",
    address = "Doha, Qatar",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D14-1086",
    doi = "10.3115/v1/D14-1086",
    pages = "787--798",
}
```