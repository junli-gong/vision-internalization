"""E5 评测框架（心智：想象必要的任务；简报 §7.4、§8.8；PREREGISTRATION.md 的 E5 草稿 v0；DEV-0018）。

现在能跑的部分：零样本基座、平凡基线、文本思维链、外部预测（LISA）的评分。E3 模型与 E5 训练接进来时沿用同一套评测集、评分与门槛。

- protocol：任务、评测集 / 开发集的序号规则、系统、标题口径与分列维度（约定的唯一来源）；
- prompts：提示配置 configs/e5_prompts.json 的读取、校验、sha256 与渲染；
- scoring：逐条解析与评分（mask IoU 两种口径、a 的起伏误差）、三列汇总、两级门槛；
- external：外部预测目录的格式、读入与输入导出（与训练数据隔离，硬约束 1）；
- runner：系统 × 模型 × 任务的运行、断点续传、metrics/compute/config/README。
入口：scripts/run_e5.py。
"""
