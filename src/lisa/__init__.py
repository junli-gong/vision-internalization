"""LISA-7B 外部对照（DEV-0075；D154；简报 §8.8 确认的 LISA 式对照系统）：在 W2 test 的 1,500 句子集上单独运行与计分。

硬约束第 1 条：LISA 把 SAM 接在 LLM 旁边，是"外接"的路线。它只作对照单独运行，输出绝不进入我们的训练数据、标签、奖励或任何挑选。
tests/test_lisa_isolation.py 静态核对：src/ 下除本包以外的代码、生成训练数据的脚本都不 import 本包或 third_party/lisa，也不引用
LISA 的结果目录。

- common.py：两个环境共用的约定（只 import 标准库与 numpy）：批准编号、目录布局、官方代码的出处核对、权重清单、协议常量与出处、
  清单与原始输出的格式。LISA 的单独环境（scripts/run_lisa.py）只 import 这一个模块。
- data.py：我们的环境里导出清单（W2 的取样，句子同 LLMSeg）。
- scoring.py：我们的环境里按 W2 的口径计分，写 W2 的叶目录。
"""
