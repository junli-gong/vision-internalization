"""Rex-Omni 外部参照（DEV-0083；D165 第四节、D177）：只做检测，在 X-M 的 xm_tune（冒烟）、xm_final、lvis_sub 上单独运行，用与 X-M 相同的
F1（src/xm/metrics）计分。两个设置：rexomni_det_joint（Rex-Omni 自己的协议，每图一次、列出全部正类别；论文里外部参照的标题读数）与
rexomni_det（每对一次、只问一个类别；与 X-M 同协议的诊断）。

许可证：官方代码与权重是 IDEA License 1.0（限非商业研究）。本包没有引入官方文件，提示、解析等是注明出处的重写；仓库公开时，按
IDEA License 第 2(b) 条加署名声明（"… is licensed under the IDEA License 1.0, Copyright (c) IDEA. All Rights Reserved."）并附许可证。

硬约束第 1 条：Rex-Omni 是学习型检测器，属于 CLAUDE.md 允许的"单独运行的对照系统"（D165 第四节点名批准，D177-c）。它的输出绝不进入我们的
训练数据、标签、奖励或任何挑选。tests/test_rexomni_isolation.py 静态核对：src/ 下除本包以外的代码、生成训练数据与训练的脚本都不 import
本包、不引用 scripts/run_rexomni.py，也不引用 Rex-Omni 的结果目录（运行目录名必须含 "_rexomni"，叶目录在 <out>/rex-omni/ 下）。

- protocol.py：官方推理的约定（只 import 标准库）：提示、图像预处理、生成参数、输出解析与坐标换算，逐条注明出处（官方仓库的文件、行号与
  commit；模型文件的 revision）。
- model.py：权重清单与核对、分词器的重建（transformers 5 读官方的慢分词器文件时把 1,022 个新增 token 的 id 排错，见 model.py）、处理器、
  真模型与 tiny 随机模型、一批生成。
- runner.py：一个划分的运行（每个"图像 × 类别"一次调用）、续跑、叶目录（同 X-M 的格式，det 用 src/xm/metrics.eval_pair，seg 为 null）。

本包只 import：标准库、numpy、PIL、torch、transformers、tokenizers、huggingface_hub、yaml，以及 src.xm.data（读对）、src.xm.metrics
（计分）、src.xm.gate（gate_xm.json 的基座那条路）、src.runlog（meta.json）、src.adapters.registry（只在 --tiny 时取分词器的来源）。
不 import 我们的训练代码、X-M 的提示与后处理（tests/test_rexomni_isolation.py 核对）。
"""
