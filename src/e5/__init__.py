"""E5 评测框架（心智：想象必要的任务；简报 §7.4、§8.8；PREREGISTRATION.md 的 E5 草稿 v0；DEV-0018）。

现在能跑的部分：零样本基座、平凡基线、文本思维链、外部预测（LISA）的评分。E3 模型与 E5 训练接进来时沿用同一套评测集、评分与门槛。

- protocol：任务、评测集 / 开发集的序号规则、系统、标题口径与分列维度（约定的唯一来源）；
- prompts：提示配置 configs/e5_prompts.json 的读取、校验、sha256 与渲染；
- scoring：逐条解析与评分（mask IoU 两种口径、a 的起伏误差）、三列汇总、两级门槛；
- external：外部预测目录的格式、读入与输入导出（与训练数据隔离，硬约束 1）；外部预测允许清单的唯一来源；
- convex_select：凸起先验基线的幅度 A 在 val 上的选定，与用冻结的 A 复算项目负责人的表（D59-4，DEV-0028）；
- runner：系统 × 模型 × 任务的运行、断点续传、metrics/compute/config/README。
入口：scripts/run_e5.py。

E5 主比较的训练与训练后评测（DEV-0037；起点 = 基座 + 预热 / E3b / E3a / 基座，在同一个 E5 训练集上训练同样的一份新 LoRA）：
- train_config：configs/e5_train.json（数据量档、epoch、学习率、提示、起伏映射、评测设置）的读取与生效参数（以 E3b 网格监督的为底）；
- token_grid：输出格式 = E3b 的像素 token 网格（二值 / 起伏 → 网格 token 的映射、允许集合、提示、解析回 E5 的格子表示）；
- train_data：train 划分的训练集（数据量档嵌套、各类等量、档内均匀、按数据种子无状态）与固定 epoch 的样本日程；
- stack：起点检查点的核对、起点 LoRA 合并进基座（再注入新 LoRA）、开训前的合并核对与权重指纹、E5 检查点的按顺序加载；
- train：训练循环（src/e3/train.Trainer 的子类）；trained_eval：评测前的合并核对与单轮受限解码的评测（与零样本同一套评分）。
入口：scripts/train_e5.py、scripts/eval_e5_trained.py。
"""
