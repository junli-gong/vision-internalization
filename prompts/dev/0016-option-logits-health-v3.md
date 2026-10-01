[DEV-TASK 0016] 适配器的"选项 logits"接口 + 健康检查 v3（MMLU-Pro 受限字母读数，D36-4）

## 背景

- 项目负责人裁定（D36-4）：健康检查里 MMLU-Pro 的标题改为"受限字母读数"，另有三点：
  - 只生成一个 token，限制在本题的选项字母上；
  - 同时记录正确选项的平均概率（连续读数对退化更敏感；标红仍按准确率）；
  - 同一字母常有带空格与不带空格两种 token，取两者中较大的 logit；
  - 改完即冻结为健康检查 v3，G3 之前不再动。
- E1 的选择题与点对任务也要用同样的受限字母读数（D36-6），由另一个任务（DEV-0017）接入。本任务提供公共接口。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `lessons.md` 的"跨机器差异"一节
- `DECISIONS.md` 的 D26、D28-7、D31、D36（第 4、6 条）
- 已合入的 `src/adapters/`（尤其 `batching.py`、`base.py`）、`src/health/`、`scripts/health_check.py`

## 任务

### 1. 适配器接口 `option_logits`

在 base 里定义，三家族实现：

```python
option_logits(inputs_list, options, *, enable_thinking) -> list[OptionScores]
```

- `options`：每条输入各自的选项列表，例如 `["A", "B", "C", "D"]`。
- 做法：
  - 与 `generate_batch` 共用前缀、助手头、思考开关与左填充；
  - 只做一次前向，取**第一个生成位置**的 logits；
  - 不经过任何采样或 logits 处理器。
- 每个选项的分数取它所有单 token 写法的 logit 最大值：
  - 写法至少包括"A"与" A"，由 tokenizer 查出；
  - 编码不是单 token 的写法不用，记下来；
  - 两种写法都不是单 token 时报错。
- 返回内容：
  - 每个选项的最大 logit 与所用 token；
  - 在选项集合上做 softmax 得到的概率；
  - argmax 选项。
  
  如果同一位置的数值与 HF 的 logits 处理器路径能对上，就写测试钉住：argmax 与 `generate(allowed_token_ids=全部写法, max_new_tokens=1)` 的结果相同。
- 精度：logits 取 fp32（从 bf16 的 lm_head 输出转换），softmax 在 fp32 下做。
- 写测试：
  - CPU 小模型上，批量与逐条结果相同；
  - 两种写法取最大值的规则；
  - 与 `generate(allowed_token_ids)` 的 argmax 一致；
  - Qwen 的思考关闭前缀（空思考块）下，"第一个生成位置"定义正确。

### 2. 健康检查 v3

- **MMLU-Pro 标题读数** `mmlu_pro_letter`：
  - 用 `option_logits`，选项为本题的实际字母；
  - 准确率为标题；
  - 另记"正确选项的平均概率"（选项集合上的 softmax）。
- **保留两个读数作诊断**：v2 的 256 token 自由作答（`mmlu_pro_free256`）与 32 token 前缀（`fmt32`）。它们不参与标红。
- GSM8K 与 IFEval 不变。
- **compare**：
  - 标红只看 `mmlu_pro_letter` 的准确率，GSM8K 与 IFEval 照旧；
  - 正确选项的平均概率给出差值与配对 bootstrap CI，作为诊断。
- 协议版本升到 3。说明里写明 D36-4，以及"G3 之前不再动"。
- 兼容 v1、v2 的输出：能读、能重评、能比较。口径不同时，照 v2 的做法写明"不可比"。

## 允许改动的文件

- `src/adapters/`：只加 `option_logits` 及其辅助函数，不改现有行为
- `src/health/`、`scripts/health_check.py`
- 对应的测试

**不许碰**：`src/e1/`（DEV-0017 在改）、`src/data/`、`src/parsing/`、`src/synth/`。

## 环境与验收

- 用 conda 环境 `visint`，不装包，不下载权重。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `bash scripts/pytest_server_parity.sh`
  - 健康检查 `--tiny --limit 3` 冒烟三家族
  - 对一对 v2 输出做 `--compare`
- 总结里写明：
  - 每个家族的字母写法与 token id；
  - 第一个生成位置的定义；
  - 与 `generate(allowed_token_ids)` 的一致性证据。
