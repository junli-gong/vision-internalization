---
license: apache-2.0
dataset_info:
- config_name: Art_Style
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 146463120.0
    num_examples: 117
  - name: test
    num_bytes: 145348441.0
    num_examples: 117
  download_size: 291074297
  dataset_size: 291811561.0
- config_name: Counting
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 4704215.0
    num_examples: 120
  - name: test
    num_bytes: 5329253.0
    num_examples: 120
  download_size: 10015874
  dataset_size: 10033468.0
- config_name: Forensic_Detection
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 19625323.0
    num_examples: 132
  - name: test
    num_bytes: 19750403.0
    num_examples: 132
  download_size: 39272509
  dataset_size: 39375726.0
- config_name: Functional_Correspondence
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 26361184.0
    num_examples: 130
  - name: test
    num_bytes: 28367706.0
    num_examples: 130
  download_size: 53227222
  dataset_size: 54728890.0
- config_name: IQ_Test
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 5306379.0
    num_examples: 150
  - name: test
    num_bytes: 4835987.0
    num_examples: 150
  download_size: 7156052
  dataset_size: 10142366.0
- config_name: Jigsaw
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 3798073.0
    num_examples: 150
  - name: test
    num_bytes: 4496412.0
    num_examples: 150
  download_size: 8085696
  dataset_size: 8294485.0
- config_name: Multi-view_Reasoning
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 9809322.0
    num_examples: 133
  - name: test
    num_bytes: 9606003.0
    num_examples: 133
  download_size: 19270001
  dataset_size: 19415325.0
- config_name: Object_Localization
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 6240389.0
    num_examples: 122
  - name: test
    num_bytes: 6441081.0
    num_examples: 125
  download_size: 12591166
  dataset_size: 12681470.0
- config_name: Relative_Depth
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 4631251.0
    num_examples: 124
  - name: test
    num_bytes: 4627481.0
    num_examples: 124
  download_size: 9203975
  dataset_size: 9258732.0
- config_name: Relative_Reflectance
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    list: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: test
    num_bytes: 18299553.0
    num_examples: 134
  - name: val
    num_bytes: 18605408.0
    num_examples: 134
  download_size: 73859421
  dataset_size: 36904961.0
- config_name: Semantic_Correspondence
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 46913152.0
    num_examples: 139
  - name: test
    num_bytes: 43943993.0
    num_examples: 140
  download_size: 90492443
  dataset_size: 90857145.0
- config_name: Spatial_Relation
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 7306558.0
    num_examples: 143
  - name: test
    num_bytes: 7472518.0
    num_examples: 143
  download_size: 14596727
  dataset_size: 14779076.0
- config_name: Visual_Correspondence
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 60403442.0
    num_examples: 172
  - name: test
    num_bytes: 56793513.0
    num_examples: 172
  download_size: 116448573
  dataset_size: 117196955.0
- config_name: Visual_Similarity
  features:
  - name: idx
    dtype: string
  - name: question
    dtype: string
  - name: sub_task
    dtype: string
  - name: image_1
    dtype: image
  - name: image_2
    dtype: image
  - name: image_3
    dtype: image
  - name: image_4
    dtype: image
  - name: choices
    sequence: string
  - name: answer
    dtype: string
  - name: prompt
    dtype: string
  - name: explanation
    dtype: string
  splits:
  - name: val
    num_bytes: 44097854.0
    num_examples: 135
  - name: test
    num_bytes: 45045115.0
    num_examples: 136
  download_size: 89068648
  dataset_size: 89142969.0
configs:
- config_name: Art_Style
  data_files:
  - split: val
    path: Art_Style/val-*
  - split: test
    path: Art_Style/test-*
- config_name: Counting
  data_files:
  - split: val
    path: Counting/val-*
  - split: test
    path: Counting/test-*
- config_name: Forensic_Detection
  data_files:
  - split: val
    path: Forensic_Detection/val-*
  - split: test
    path: Forensic_Detection/test-*
- config_name: Functional_Correspondence
  data_files:
  - split: val
    path: Functional_Correspondence/val-*
  - split: test
    path: Functional_Correspondence/test-*
- config_name: IQ_Test
  data_files:
  - split: val
    path: IQ_Test/val-*
  - split: test
    path: IQ_Test/test-*
- config_name: Jigsaw
  data_files:
  - split: val
    path: Jigsaw/val-*
  - split: test
    path: Jigsaw/test-*
- config_name: Multi-view_Reasoning
  data_files:
  - split: val
    path: Multi-view_Reasoning/val-*
  - split: test
    path: Multi-view_Reasoning/test-*
- config_name: Object_Localization
  data_files:
  - split: val
    path: Object_Localization/val-*
  - split: test
    path: Object_Localization/test-*
- config_name: Relative_Depth
  data_files:
  - split: val
    path: Relative_Depth/val-*
  - split: test
    path: Relative_Depth/test-*
- config_name: Relative_Reflectance
  data_files:
  - split: val
    path: Relative_Reflectance/val-*
  - split: test
    path: Relative_Reflectance/test-*
- config_name: Semantic_Correspondence
  data_files:
  - split: val
    path: Semantic_Correspondence/val-*
  - split: test
    path: Semantic_Correspondence/test-*
- config_name: Spatial_Relation
  data_files:
  - split: val
    path: Spatial_Relation/val-*
  - split: test
    path: Spatial_Relation/test-*
- config_name: Visual_Correspondence
  data_files:
  - split: val
    path: Visual_Correspondence/val-*
  - split: test
    path: Visual_Correspondence/test-*
- config_name: Visual_Similarity
  data_files:
  - split: val
    path: Visual_Similarity/val-*
  - split: test
    path: Visual_Similarity/test-*
---

# BLINK: Multimodal Large Language Models Can See but Not Perceive

[**🌐 Homepage**](https://zeyofu.github.io/blink/) | [**💻 Code**](https://github.com/zeyofu/BLINK_Benchmark) | [**📖 Paper**](https://arxiv.org/abs/2404.12390.pdf) | [**📖 arXiv**](https://arxiv.org/abs/2404.12390) | [**🔗 Eval AI**](https://eval.ai/web/challenges/challenge-page/2287/overview)


This page contains the benchmark dataset for the paper "[BLINK: Multimodal Large Language Models Can See but Not Perceive](https://arxiv.org/abs/2404.12390.pdf)"


## Introduction
We introduce **BLINK**, a new benchmark for multimodal language models (LLMs) that focuses on core visual perception abilities not found in other evaluations. Most of the **BLINK** tasks can be solved by humans “within a blink” (e.g., *relative depth estimation, visual correspondence, forensics detection, and multi-view reasoning*). However, we find these perception-demanding tasks cast significant challenges for current multimodal LLMs because they resist mediation through natural language. **BLINK** reformats 14 classic computer vision tasks into 3,807 multiple-choice questions, paired with single or multiple images and visual prompting. While humans get 95.70% accuracy on average, **BLINK** is surprisingly challenging for existing multimodal LLMs: even the best-performing GPT-4V and Gemini achieve accuracies of 51.26% and 45.72%, only 13.17% and 7.63% higher than random guessing, indicating that such perception abilities have not “emerged” yet in recent multimodal LLMs. Our analysis also highlights that specialist CV models could solve these problems much better, suggesting potential pathways for future improvements. We believe **BLINK** will stimulate the community to help multimodal LLMs catch up with human-level visual perception.

![Alt text](assets/teaser.png)
## Dataset Creation

BLINK is created to challenge multimodal models on hollistic visual perception abilities with tasks inherited from classic computer vision problems, stimulating future development of multimodal LLMs that achieve human-level visual perception. 
**Unique Features** of BLINK includes diverse visual prompting, beyond recognition perception, and visual commonsense.

<!-- ![Alt text](assets/comparison.png) -->
## Load Dataset
```
import datasets

dataset_name = 'BLINK-Benchmark/BLINK'
data = load_dataset(dataset_name, SUBTASK_NAME)
```
where `SUBTASK_NAME` is one of the subtasks: `['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation']`


## 🏆 Mini-Leaderboard
| Model                      | Val (1,901) | Test (1,907) |
|----------------------------|:-----------:|:------------:|
|🏅 Human                    |     95.7    |     95.7     |
|🥈 GPT-4o                   |     60.0    |   **59.0**   |
|🥉 GPT-4 Turbo              |     54.6    |     53.9     |
| GPT-4V(ision preview)    |     51.1    |     51.3     |
| Gemini Pro               |     45.2    |     45.7     |
| LLaVA-1.6-34B              |     46.8    |     45.1     |
| Claude 3 Opus              |     44.1    |     44.1     |
| Yi-VL-34B                  |     41.7    |     42.8     |
| Qwen-VL-MAX                |     40.3    |     42.0     |
| LLaVA-v1.5-13B-xtuner      |     42.0    |     41.3     |
| Yi-VL-6B                   |     38.7    |     41.7     |
| LLaVA-v1.5-7B-xtuner       |     39.4    |     40.8     |
| LLaVA-1.5-13B              |     42.7    |     40.6     |
| InstructBLIP-13B           |     42.2    |     39.6     |
| CogVLM                     |     41.5    |     39.4     |
| InstructBLIP-7B            |     39.7    |     38.7     |
| OpenFlamingo2-9B           |     39.2    |     38.3     |
|👀 **Random Choice**          |     38.1    |     38.1     |
| LLaVA-1.5-7B               |     37.1    |     38.0     |
| LLaVA-internLM2-7B         |     37.7    |     36.1     |
| MiniGPT-4-v2-7B            |     34.2    |     34.6     |

<img src="assets/radar_v1.png" width="400" />

🎯 **We have released a full suite comprising 1,901 validation samples, the prompts we used, and [model predictions](https://github.com/zeyofu/BLINK_Benchmark/tree/main/eval) for the baselines tested in our paper. However, the 1,907 test questions are available without their answers.** You can submit your model's predictions for the **test set** on **[EvalAI](https://eval.ai/web/challenges/challenge-page/2287/overview)**.

## Disclaimers
Blink makes use of data from existing image datasets, and does not cover all the visual perception abilities in the wild. For the forensics detection task, we manually collected images that are publicly available from online search. We have made every effort to ensure that the images included in this paper are used in accordance with applicable copyright laws and are properly credited. However, if you are the copyright owner of any image included in our work and believe that its use conflicts with your licensing agreements, please [contact](#contact) us directly. We are committed to addressing any legitimate concerns promptly.

## Contact
- Xingyu Fu: xingyuf2@seas.upenn.edu
- Yushi Hu:  yushihu@uw.edu
- Wei-Chiu Ma:    weichiu@cornell.edu
- Ranjay Krishna: ​ranjay@cs.washington.edu

## Citation

**BibTeX:**
```bibtex
@article{fu2024blink,
  title={BLINK: Multimodal Large Language Models Can See but Not Perceive},
  author={Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
  journal={arXiv preprint arXiv:2404.12390},
  year={2024}
}
```
Daily Paper: https://huggingface.co/papers/2404.12390