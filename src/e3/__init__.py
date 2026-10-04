"""E3b：像素 token 通道（简报 §8.5；DEV-0025 第一部分：词表、目标、数据管线、训练循环、C3 对照；DEV-0030 第二部分：评测）。

模块：
- tokens：网格词表（2 + 64 + 1 = 67 个新 token）、预算核对、加进适配器之后的 id 布局与受限 logits 的允许集合；
- targets：GT → 32×32 网格 → token 序列（光栅顺序、行分隔、停止 token，长度 1057）与严格解析；深度的无效像素处理；
- feedback：叠加反馈图（训练的修正对与评测的叠加反馈迭代共用的唯一渲染入口）；
- depth_noise：深度网格的加噪过程（分割用 src/noising/mask_noise）；
- c3：C3 对照的稀疏文本目标（框用本家族原生格式与 E1 的定位提示、多边形、相对深度陈述）；
- data：训练池（去重）、无状态的样本日程、修正对与 C3 样本的构造；
- config：configs/e3b_train.json 的读取与生效参数；
- train：训练循环（LoRA + 网格 token 行、受限交叉熵 / 分块交叉熵、AdamW、warmup + 余弦、检查点与续跑、记账）；另有 C1′（变体 c1prime：
  冻结 LLM、只训练网格 token 的行，DEV-0035 §1）与网格格式预热（监督 warmup，DEV-0035 §2）。
- warmup：网格格式预热的样本（纯文字给坐标 → 网格，不给图像）、固定的留出集、格式错误率与停止规则。
- evaluate：评测（单轮、叠加反馈多轮与计算记账、匹配计算量的单轮、汇总与输出）；eval_scoring：网格的紧凑编码、上采样到原图、
  逐样本指标；merge_check：评测前的合并核对（scripts/check_training_interfaces.py --checkpoint 的同一套函数）。
入口：scripts/train_e3b.py（训练）、scripts/eval_e3b.py（评测）；训练后模型的 T 文本评测是 scripts/run_e1.py --checkpoint。
各模块的规则写在各自的模块说明里。
"""
