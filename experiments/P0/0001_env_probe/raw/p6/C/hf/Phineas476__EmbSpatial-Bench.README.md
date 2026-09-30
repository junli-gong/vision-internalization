---
license: cc-by-4.0
---
### EmbSpatial-Bench 

EmbSpatial-Bench is a benchmark for evaluating embodied spatial understanding of LVLMs. The benchmark is automatically derived from embodied scenes and covers 6 spatial relationships from an egocentric perspective.
The constructed benchmark comprises a total of 3,640 QA pairs, covering 294 object categories and 6 spatial relationships.

### EmbSpatial-SFT 

EmbSpatial-SFT is an instruction-tuning dataset, which provides QA data for two tasks: spatial relationship identification and object localization. 
The first task setting is consistent with EmbSpatial-Bench, while the other serves as an auxiliary task to enhance the model’s ability to ground target objects. The auxiliary task can be considered as the foundational skill for relationship identification. 
EmbSpatial-SFT is solely built on the training scenes of MP3D. 

More details can be seen in our paper: https://arxiv.org/abs/2406.05756.