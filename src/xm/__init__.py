"""X-M（D165；docs/lead_answers/2026-11-03-xm-directive.md）：一个 LLM 同时做类别提示的检测与实例分割。

- src/xm/data.py：数据接口（XMInstance、XMPair、load_pairs）、三个 COCO val2017 评测划分与密封、LVIS 子集、ReasonSeg val（DEV-0079）；
- src/xm/exclusions.py：训练的去重清单 configs/xm_train_exclusions.json（DEV-0079）；
- src/xm/targets.py：训练目标（框列表 + 每实例 B 样条）、阅读顺序、分片与 manifest（DEV-0079）。
"""
