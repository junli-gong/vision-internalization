"""E3b：像素 token 通道（简报 §8.5；DEV-0025 第一部分：词表、目标、数据管线、训练循环、C3 对照；评测是第二部分）。

模块：
- tokens：网格词表（2 + 64 + 1 = 67 个新 token）、预算核对、加进适配器之后的 id 布局与受限 logits 的允许集合；
- targets：GT → 32×32 网格 → token 序列（光栅顺序、行分隔、停止 token，长度 1057）与严格解析；深度的无效像素处理；
- feedback：叠加反馈图（训练的修正对与评测的叠加反馈迭代共用的唯一渲染入口）；
- depth_noise：深度网格的加噪过程（分割用 src/noising/mask_noise）；
- c3：C3 对照的稀疏文本目标（框、多边形、相对深度陈述）；
- data：训练池（去重）、无状态的样本日程、修正对与 C3 样本的构造；
- config：configs/e3b_train.json 的读取与生效参数；
- train：训练循环（LoRA + 网格 token 行、受限交叉熵 / 分块交叉熵、AdamW、warmup + 余弦、检查点与续跑、记账）。
入口：scripts/train_e3b.py。各模块的规则写在各自的模块说明里。
"""
