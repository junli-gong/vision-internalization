---
language:
- en
dataset_info:
  features:
  - name: image
    dtype: image
  - name: text
    dtype: string
  - name: question
    dtype: string
  - name: number
    dtype: int32
  splits:
  - name: test
    num_bytes: 23541580.0
    num_examples: 491
  download_size: 23385624
  dataset_size: 23541580.0
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

This dataset was introduced in PaliGemma for evaluating counting in vision language models. This version only includes 491 images from the original CountBench dataset, since some of the original URLs can no longer be accessed.

### Original Description

* CountBench: We introduce a new object counting benchmark called CountBench,
  automatically curated (and manually verified) from the publicly available
  LAION-400M image-text dataset. CountBench contains a total of 540 images
  containing between two and ten instances of a particular object, where their
  corresponding captions reflect this number.

* CountBenchQA: Each image is paired with a manually generated question about the
  number of objects in the image to turn CountBench into a VQA task.

```
@article{beyer2024paligemma,
      title={{PaliGemma: A versatile 3B VLM for transfer}},
      author={Lucas Beyer and Andreas Steiner and André Susano Pinto and Alexander Kolesnikov and Xiao Wang and Daniel Salz and Maxim Neumann and Ibrahim Alabdulmohsin and Michael Tschannen and Emanuele Bugliarello and Thomas Unterthiner and Daniel Keysers and Skanda Koppula and Fangyu Liu and Adam Grycner and Alexey Gritsenko and Neil Houlsby and Manoj Kumar and Keran Rong and Julian Eisenschlos and Rishabh Kabra and Matthias Bauer and Matko Bošnjak and Xi Chen and Matthias Minderer and Paul Voigtlaender and Ioana Bica and Ivana Balazevic and Joan Puigcerver and Pinelopi Papalampidi and Olivier Henaff and Xi Xiong and Radu Soricut and Jeremiah Harmsen and Xiaohua Zhai},
      year={2024},
      journal={arXiv preprint arXiv:2407.07726}
}

@article{paiss2023countclip,
      title={{Teaching CLIP to Count to Ten}},
      author={Paiss, Roni and Ephrat, Ariel and Tov, Omer and Zada, Shiran and Mosseri, Inbar and Irani, Michal and Dekel, Tali},
      year={2023},
      journal={arXiv preprint arXiv:2302.12066}
}
```