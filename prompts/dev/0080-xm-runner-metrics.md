[DEV-TASK 0080] X-M 的推理与评测：两次调用的提示与解析、NMS、逐实例三元图贴边、对照设置、基于匹配的 F1、密封守卫

## 背景

- 负责人新开 X-M 线（D165，要点原文 `docs/lead_answers/2026-11-03-xm-directive.md`，先读全文，尤其第三、四、五节）。
  - 一个 LLM 做类别提示的检测与实例分割，做法同 X-2：LLM 先给框，再描 16 个控制点的 B 样条，GrabCut 三元图贴边，不含学习的工具，不用任何学习型视觉模型。
- 数据层由 DEV-0079 并行做（`src/xm/data.py`，接口见下）；训练由 DEV-0081 随后做。这里只做推理、后处理、评测与指标。
- 开发机没有 COCO 数据与 GPU：用 `tests/fixtures` 的小数据与 tiny 模型测，真跑在 HPC。

## 开工前（省盘）

开发机磁盘紧（约 12 GB，另一个开发 agent 同时在跑）。开工前在**你自己的 worktree 里**执行 `rm -rf experiments/`：本任务用不到已有结果，可省约 6 GB。不要 `git add` 这些删除，也不要动主仓库。只跑你自己的测试与 X-2 的相关测试。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `docs/lead_answers/2026-11-03-xm-directive.md`、`DECISIONS.md` 的 D165
- X-2 的运行器与工具：`scripts/run_x2.py`、`src/explore/x2_runner.py`、`x2_prompts.py`、`x2_tools.py`、`x2_render.py`、`src/explore/guidance.py`（样条光栅化、三元图、GrabCut、框内椭圆）
- 密封的写法：`src/data/explore_splits.py`（G-EX 守卫）、`scripts/run_x2.py` 对 test 的放行（DEV-0072）
- E1 的出框协议：`scripts/run_e1.py`、`src/e1/`（基座零样本的图像口径、解码、解析）
- W2 的计分与叶目录格式：`src/w2/`、`scripts/mgr/w2_test_table.py`（`preds`、`metrics`）

## 数据接口（DEV-0079 在 `src/xm/data.py` 实现；你按这个接口写，测试里用自己造的 fixture）

```python
@dataclass(frozen=True)
class XMInstance:
    ann_id: int; box_xyxy: tuple[float, float, float, float]; rle: dict; area: int

@dataclass(frozen=True)
class XMPair:
    id: str; source: str; split: str; image: str; width: int; height: int
    category_id: int; category: str
    instances: tuple[XMInstance, ...]; crowd_rles: tuple[dict, ...]
```

`load_pairs(split, data_root)`。`split` 取值：`xm_tune`、`xm_decision`（密封，门名 G-XM）、`xm_final`、`lvis_sub`、`reasonseg_val`、`train`。DEV-0079 未合入之前，你在 `src/xm/` 里放一个同名接口的最小桩，合入时以 DEV-0079 的为准。

## 任务

1. **提示与解析**（`src/xm/prompts.py`），两次调用，与 X-2 一致的 0–1000 网格整数 JSON：
   - 调用 1（列出实例）：给图像与类别名，要模型按阅读顺序输出该类别所有实例的框 `[[x1,y1,x2,y2], …]`，最多 30 个；没有就输出 `[]`。
   - 调用 2（描轮廓）：给图像、类别名与调用 1 解析出的框（按编号列在提示里），要模型为每个框输出一条 16 个控制点的闭合 B 样条，`[[[x,y]×16], …]`，与框一一对应。
   - 解析：严格 JSON。
     - 框多于 30 个只取前 30 个；坐标越界就截到网格内。
     - 调用 2 的条数与框数不符、或某条不是 16 个点时，该实例记"样条失败"，按规则退回框内椭圆（同 X-2 的退回），并记数。
   - 提示词放成常量，调参集 `xm_tune` 上可以改；改动要进运行身份。
2. **确定性的后处理**（`src/xm/post.py`）：
   - 框的 NMS：IoU 阈值做成常量，缺省 0.7，管理员在预注册里定；
   - 控制点限制在各自框内（同 X-2）；
   - 逐实例光栅化，作 GrabCut 三元图的种子贴边。三元图的带宽等参数照 X-2 的 `x2_spline_trimap`，每个实例独立贴边，允许重叠。
3. **设置**（`scripts/run_xm.py --setting …`，可一次跑多个，共用模型输出）：
   - `xm_spline_trimap`：主设置，X-M 的框 + X-M 的样条 → 三元图贴边；
   - `xm_box_grabcut`：X-M 的框直接交给 GrabCut，不描样条，用 W1-F 的缺省 GrabCut 候选（同 W2 的 `gc_default`）。这是判定第 1 条的对照；
   - `xm_ellipse_trimap`：X-M 的框换成框内椭圆，再三元图贴边（同 X-2 的椭圆消融）；
   - `xm_spline_raster`：诊断，样条直接光栅化，不贴边；
   - `base_det`：基座零样本。同一个调用 1 的提示，E1 的图像口径与解码，不加 LoRA。只出框，是判定第 2 条的对照；
   - `base_box_grabcut`：基座零样本的框交给缺省 GrabCut，作实例分割的起点。
   - 检测的读数对 `xm_*` 各设置相同，都用 X-M 调用 1 的框，只算一次。
4. **指标**（`src/xm/metrics.py`；LLM 不给置信度，用基于匹配的 F1，与 Rex-Omni 一致）：
   - 每个对内：预测与真值按 IoU 一对一匹配，IoU ≥ 0.5 记 TP。
     - 匹配算法、多对一的处理做成可选常量：缺省按 IoU 从高到低贪心一对一，可选匈牙利。
     - 没匹配上的预测，如果与该对某个 crowd 区域的 IoA ≥ 0.5 就忽略，不记 FP（COCO 惯例）；其余记 FP。没匹配上的真值记 FN。
   - 检测用框 IoU，实例分割用 mask IoU。
   - 数据集读数：
     - F1 = 2ΣTP / (2ΣTP + ΣFP + ΣFN)，各对求和；
     - 另报 precision、recall，以及匹配上的平均 IoU 与平均 mask IoU。
   - 置信区间与配对差：按**图像**重抽（同一张图的多个类别一起抽），95%，10,000 次，种子 `src/constants.BOOTSTRAP_SEED`。写成函数，供判定脚本调用。
   - 指代分割（ReasonSeg）：gIoU 与 cIoU，同 W2 口径。
5. **运行器 `scripts/run_xm.py`**：
   - `--model`、`--checkpoint`（`base_*` 设置不配检查点）、`--split`、`--setting …`、`--data-root`、`--out`、`--limit`、`--resume`、`--tiny`，`--candidate-workers` / `--tool-workers` 同 run_x2。
   - 输出叶目录 `<out>/<模型>/xm/<设置>/<划分>/`：
     - predictions.jsonl：每对一行，含 id、原始输出、解析出的框与样条、每个实例的 mask RLE、TP/FP/FN、匹配上的 IoU、失败与退回；
     - metrics.json：F1、P、R、平均 IoU、CI 用 1,000 次的运行时读数、失败计数、complete；
     - compute.json：调用次数、token、用时。
   - `--out` 的末级目录名必须以四位运行编号开头。
   - **密封守卫**：
     - `xm_decision` 必须给 `--gate G-XM`，否则拒绝，退出码 2，什么都不写；
     - `xm_final` 与 `lvis_sub` 必须给 `--xm-gate <gate_xm.json>`，内容为"判定通过"、非彩排、判定点全真、检查点 manifest 相同，否则拒绝。gate_xm.json 由 DEV-0081 的判定脚本写，这里按约定的键读：`verdict`、`rehearsal`、`checks`、`candidate.manifest_sha256`；
     - `reasonseg_val`：对 X-M 检查点同样要 `--xm-gate`；`--system x2`（X-2 的检查点做指代分割，指令第六节 D1–D3 的描述性运行）不要。
6. **ReasonSeg 上的指代分割**：
   - `--system x2`：用 X-2 的检查点按 X-2 原协议（`x2_spline_trimap`）跑；
   - `--system xm`：用 X-M 的检查点同样跑。
   - 指令原则说指代分割仍按 X-2 原协议。复用 `src/explore/x2_runner.py` 的函数，不另写。
7. **测试**（`HF_HUB_OFFLINE=1 /opt/miniconda3/envs/visint/bin/python -m pytest tests/ -q -k "xm"`，另跑 `tests/test_x2_*.py` 确认 X-2 不变）：
   - 解析：正常、空、越界、超过 30、条数不符、退回；
   - NMS；
   - 指标：手算的小例子，覆盖贪心与匈牙利、crowd 忽略、空预测、空真值；
   - 按图像重抽的 bootstrap 与 `src.stats` 的参考实现一致；
   - tiny 上各设置都出叶目录；
   - 密封：缺门名、缺 gate、gate 不合格时拒绝；
   - `--out` 不带运行编号时拒绝；
   - X-2 的输出逐字节不变。
8. 在总结里给出服务器命令（06/07、调参集与判定集、各设置）与每对的用时估计（按 X-2 的实测折算，写明依据）。

## 不做

- 不写数据层（DEV-0079）与训练（DEV-0081）。
- 不改 X-2 的任何代码、配置与结果。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 提示词全文；
- 指标的定义；
- 服务器命令与用时估计；
- 测试结果原文；
- 待裁的地方。

尽量 3 小时内完成。
