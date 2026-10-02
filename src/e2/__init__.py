"""E2：探针——稠密视觉知识在不在 LLM 隐藏状态里（简报 §8.3；DEV-0023）。

模型冻结（no_grad、bf16），在图像 token 位置对一组层在线训练逐层探针，不落盘缓存特征（简报 §4 第 8 条、§5.3）：

- 输入（src/e2/features.py）：图像口径沿用 E1 冻结的（D28-3）。深度、语义分割、法向用"图在前"，提示是 configs/e2_probes.json
  里一句与任务无关的话；指代分割用"指令在前"，指代句在图像之前（图像 token 能看到它，检验任务调制）。
- 特征：一次 hidden_states 前向取全部选定层在图像位置的隐藏状态，按适配器的 ImageTokenInfo.grid_hw 排成 [H_t, W_t, D]
  （行主序，src/e2/grid.py），转 fp32，按通道标准化（统计量训练前定、之后冻结）。同一批特征喂给所有层的探针。
- 探针（src/e2/probes.py）：每个（层, 种类）一个探针、一个优化器。线性 = 1×1 卷积 D → C；两层卷积 = 3×3 D → 256、GELU、3×3 → C。
  输出在 token 网格分辨率，双线性上采样到 GT 分辨率后算损失与指标（src/e2/tasks.py）。
- 任务：depth（NYU，SILog）、semseg（ADE20K，ADE_NUM_CLASSES 类，交叉熵）、normals（NYU 法向，余弦损失）、refseg（RefCOCO，BCE + Dice）。
- 层：--layers all / every k，按 num_layers() 展开（层号约定见 src/adapters/base.py）；另加 L{N}_pre（最后一个 decoder 层的原始输出，
  最终归一化之前；适配器的层键 PRE_NORM_LAST；DEV-0024，D53-3 的 I1），单列。参考线：post_proj（= 第 0 层，D21-3）、
  pre_proj（视觉塔投影 / 合并之前的输出，需要适配器提供 pre_projection_features，现有适配器没有）。
- 评测另按光栅位置分开报标题指标（上半 / 下半、token 光栅下标的四分位；src/e2/positions.py；D53-3 的 I2），点估计。
- 去重：训练池的剔除清单必须比较过当前任务的评测集（src/e2/data.check_dedupe_coverage；D53-3 的 B）。
- 训练：固定种子、步数、学习率与调度（配置里，全部模型与层共用）；每 eval_every 步在训练集留出的小验证子集上记曲线；最后在
  完整评测集上评一次，带 bootstrap CI。不按评测集挑步数或超参。
- 输出与续跑（src/e2/runner.py）：experiments/E2/<编号>/<模型>/<任务>/，检查点保存探针、优化器、调度器与标准化统计量。

入口：scripts/run_e2.py。
"""
