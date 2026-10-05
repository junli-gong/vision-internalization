"""E3a：D 通道的稠密读出（简报 §3.3、§7.3、§8.4；硬约束第 3 条；DEV-0034 第一部分：框架、C1/C2 对照、泄漏检查、训练与评测）。

**结构**（model.py；超参在 configs/e3a_train.json，正式数值由 E3 预注册冻结）：
- 查询 token：16 × 16 = 256 个可学习的嵌入（fp32 主副本，前向时转成模型 dtype），指令在前，查询 token 追加在指令与图像之后
  （适配器的 inject_query_tokens；位置模式按家族取配置：Qwen grid2d，其余 seq1d）。
- 逐层加权混合（LayerMix）：全部 LLM 层（层号 0..num_layers()，D19 的约定：含第 0 层，最后一层是归一化之后）在查询位置的隐藏状态，
  每层先做无参数的 LayerNorm（逐 token，fp32），再按 softmax(logits) 加权求和、乘 gamma（ELMo 的 scalar mix）。混合权重参与训练，
  训练前后的权重写进 train_summary.json，每步写进 train_log.jsonl。
- 共享薄解码器（ThinDecoder）：[16, 16, hidden] → 128 × 128 × 64 的逐像素嵌入场；LayerNorm + 线性投影到 256 通道，再接卷积块与
  固定的 2× 双线性上采样（16 → 32 → 64 → 128）。参数量上限 DECODER_MAX_PARAMS（50M），构造时断言并报告实际数量。
- 输出头：语义分割 = 开放词汇（类别名在同一条提示里列出，类别名 token 的混合隐藏状态取平均 → class_proj 线性投影到 64 维，与像素
  嵌入做点积）；指代分割、深度、法向 = 各一个 1×1 线性适配（64 → 1 / 1 / 3），每个不超过 HEAD_MAX_PARAMS（1K），构造时断言。
- 硬约束第 3 条（解码器薄、看不到像素）：读出模块的 forward 只收 LLM 的隐藏状态（查询位置与类别名位置）；构造时核对输入来源、
  子模块类型与参数（check_no_pixel_path）；tests 另用梯度检查钉住"解码器输出对像素张量的梯度只经过 LLM 路径"。唯一的像素参与是
  评测时的无学习上采样（evaluate.py：以原图为引导图的 guided，另报 bilinear）。

**变体**（同一个训练脚本，--variant）：
- full：LoRA（r 32，LLM 全部线性层，视觉塔冻结）+ 查询嵌入 + 混合权重 + 解码器与各头；
- c1：冻结 LLM（没有 LoRA），只训练查询嵌入、混合权重、解码器与各头；
- c2：解码器直接读 vision_features（投影之后、进入 LLM 之前的图像 token 网格），无学习的双线性（抗混叠）缩放到 16 × 16，接同一个
  解码器；没有 LoRA、没有查询 token、没有混合权重（只有一个来源）。类别名的表示取 LLM 的输入嵌入（第 0 层，不经过任何 decoder 层）。

**模块**：config（配置的读取、核对与生效参数）、model（读出结构、参数量断言、像素通路的结构检查）、targets（GT → 128 × 128：
标签最近邻，深度与法向按有效像素的面积平均）、losses（同 E2 的四种损失，在 128 × 128 上）、data（训练池、无状态的样本日程、样本构造）、
features（LLM 的查询 / 类别名隐藏状态，C2 的视觉特征）、train（训练循环、检查点、续跑、记账）、evaluate（稠密指标、两种上采样、
泄漏检查、逐样本统计与配对比较）。入口：scripts/train_e3a.py、scripts/eval_e3a.py。各模块的规则写在各自的模块说明里。
"""
