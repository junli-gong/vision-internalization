# W1（仪器工作台）开发规划与接口约定（管理员，2026-10-20；依据 D103，原文 `docs/lead_answers/2026-10-20-w1-directive.md`）

W1：LLM 保持冻结，只给不含学习的通用工具，在多轮回路里完成高层 CV 任务。本文件是三个开发任务（DEV-0049、0050、0051）的共同接口约定。三个任务并行开发，各自只改自己的部分，按这里的签名对接；需要改约定时写进完成总结，由管理员定。

## 0. 硬约束（W1，D103）

- **不使用任何学习型视觉模型或预训练权重**：SAM、检测器、深度模型、CLIP、DINO、OCR 模型都不行，也不能用 OpenCV 的 DNN 模块。
  - 工具与沙箱里只允许不含学习的算法。
  - 所有任务共用同一套工具与回路。
- **LLM 冻结**：不加载任何 LoRA，不训练。
  - 语言能力按构造不变；健康检查照报，并注明与基座逐位相同。
- **工作台与任务无关**：按任务写的提示词只说明任务是什么、输出成什么格式，不写解题方法。系统提示与工具说明对所有任务完全相同。
- **计算记账**（简报硬约束第 11 条）：每条样本记 token 数（提示、生成、补入）、工具调用次数、轮数与墙钟。

## 1. 适配器的多轮对话接口（DEV-0049，`src/adapters/`）

新增，不改现有 API 的行为；只实现 Qwen3.5 家族，其余家族抛 `NotImplementedError`。

```python
# src/adapters/chat.py（新）
@dataclass(frozen=True)
class ToolSpec:            # 一个工具的说明，按 Qwen3.5 对话模板的 tools 参数渲染
    name: str
    description: str
    parameters: dict       # JSON Schema（type=object，properties，required）

@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]   # 模板的 <parameter=…> 都是字符串；由调用方按 schema 解析
    raw: str                    # 原文，含 <tool_call> 标记

# 消息：{"role": "system"|"user"|"assistant"|"tool", "content": [{"type": "text", "text": str} | {"type": "image", "image": PIL.Image}]}
# assistant 消息可带 "tool_calls": [ToolCall]（历史回填用）；tool 消息的 content 可以含图像（模板渲染为 <tool_response>）。

@dataclass(frozen=True)
class ChatTurnResult:
    text: str                  # 作答段原文（</think> 之后，不含停止 token）
    thinking: str              # 思考段原文
    thinking_end: str          # natural | forced | unclosed | not_opened（同 GenerationResult）
    tool_calls: list[ToolCall] # 从作答段解析出的 <tool_call>，按出现顺序；解析失败的记进 tool_call_errors
    tool_call_errors: list[str]
    stopped: bool              # 以停止 token 结束；False = 作答段被截断
    n_prompt_tokens: int
    n_generated_tokens: int
    n_inserted_tokens: int     # 强制收尾补入的 token
    n_visual_tokens: int
    n_images: int
    wall_time_s: float

class ModelAdapter:            # base.py 新增方法；Qwen3.5 实现
    def chat_turn_batch(self, conversations: Sequence[Sequence[dict]], *, tools: Sequence[Sequence[ToolSpec] | None],
                        thinking_budget: int, answer_budget: int) -> list[ChatTurnResult]: ...
    def chat_turn(self, messages, *, tools, thinking_budget, answer_budget) -> ChatTurnResult:   # = 批大小 1
        ...
```

- **按轮批量**：
  - 吞吐是关键。9B 单条解码约每秒几十个 token，一条样本最多 6 轮 × 2,048 个思考 token，逐条跑开发-检查集要几十小时。
  - 所以回路按轮同步：每一轮，所有还没结束的样本拼成一批，左填充，一次生成。
  - 与 `generate_batch_budgeted` 相同，批内每行各自强制收尾。
  - 批内结果与逐条跑的结果逐 token 相同（贪心；同 E1 的批一致性规则，D36-7）。

- 开思考；思考段到 `thinking_budget` 就强制收尾，与 `generate_batch_budgeted` 的规则相同（budget.py）。
- 贪心解码，确定。
- 每张图都按适配器的图像策略处理（默认 `PixelBudget(300)`），包括后面几轮的新图。
- 历史里的 assistant 思考段按模板的规则处理，模板会去掉最后一个 user 之前的思考段。

## 2. 工具台与沙箱（DEV-0050，`src/w1/`）

```python
# src/w1/workbench.py（新）
GROUPS = ("eye", "hand", "snap", "check", "code", "submit")   # 消融按组关掉：A1 = 去掉 eye；A2 = 去掉 check；A3 = 去掉 snap 与 code

@dataclass
class ToolResult:
    text: str                       # 给模型看的文字结果（坐标、读数、报错）
    images: list["PIL.Image.Image"] # 给模型看的图（放大视图、叠加图）
    error: str | None               # 参数错误、越界、沙箱报错；error 时 text 写明原因

class Workbench:
    def __init__(self, image: "PIL.Image.Image", *, disabled_groups: frozenset[str] = frozenset(),
                 sandbox: "SandboxConfig | None" = None): ...
    def specs(self) -> list[ToolSpec]          # 当前可用的工具（已去掉 disabled 组）
    def call(self, name: str, arguments: dict[str, str]) -> ToolResult
    @property
    def submission(self) -> "Submission | None"   # submit_* 工具写入：mask（原图分辨率 bool 数组）、box（绝对 xyxy）或 count（int）
    def log(self) -> list[dict]                # 每次调用：名字、参数、耗时、是否出错、沙箱代码原文
```

- **坐标约定**：所有工具的坐标都用整张原图的 0–1000 相对坐标，x 在前，与 Qwen 的原生框格式一致。放大视图也用原图坐标，返回的文字写明视图覆盖的范围。
- **状态**：
  - 命名 mask（原图分辨率）；
  - 当前图像注册表（id → 图）。
  - `submit_mask(name)`、`submit_box(...)`、`submit_count(n)` 写最终答案。
- **沙箱**（`code` 组，`run_python`）：
  - 子进程，`python -I`，空环境变量，临时工作目录，限时与限内存。
  - import 白名单：numpy、scipy、cv2（屏蔽 `cv2.dnn`、`cv2.face`、`cv2.text` 与带预训练模型的类，例如 `CascadeClassifier`、默认行人检测器）、skimage（屏蔽 `skimage.feature.Cascade` 与需要下载数据的模块）、math、itertools、collections、json、re。
  - 不能打开文件，不能联网。
  - 输入：当前图像与命名 mask。输出：stdout 文字，可选返回新的 mask。
  - 每段代码原文进 `log()`。
- 工具的完整清单与参数由 DEV-0050 定，写进 `docs/w1_tools.md`；DEV-0051 只通过 `specs()` / `call()` / `submission` 使用。

## 3. 回路、任务、评分、基线（DEV-0051，`src/w1/` 与 `scripts/run_w1.py`）

```python
# src/w1/loop.py（新）
@dataclass(frozen=True)
class Protocol:          # configs/w1.json；在开发-调参集上定，跑 test 之前冻结（D103 §二.4）
    max_rounds: int = 6
    max_tool_calls: int = 8
    thinking_budget: int = 2048
    answer_budget: int = 1024
    disabled_groups: frozenset[str] = frozenset()

def run_episodes(adapter, samples, task: str, protocol: Protocol, prompts, *, batch_size: int) -> list["EpisodeRecord"]: ...
# 按轮同步：每轮把还没结束的样本拼成一批调 adapter.chat_turn_batch，再逐条执行工具调用、追加 tool 消息；结束的样本退出批
```

- **任务**：
  - refseg：指代分割，主任务，cIoU、gIoU；
  - refexp_box：定位，Acc@0.5；
  - count：CountBench，准确率。
  - 读数与评分用 E1 的函数（`src/e1/scoring`、`src/metrics/seg`、`src/metrics/box`）。
- **最终答案**：
  - mask 由 `submit_mask` 提交（工具光栅化）；
  - 框用原生格式提交（`submit_box`，或作答段里的原生 JSON，由 E1 的解析器读）；
  - 计数提交一个整数。
- **数据划分**：
  - 开发-调参 300 句与开发-检查 500 句，都从 `poly:refcoco_unc_val`（1000 句）里按 `seeded_sample` 不重叠地取，salt 新定；
  - test 用 E1 的 `poly:<split>` 子集，每个划分 1000 句：refcoco testA/testB、refcoco+ testA/testB、refcocog test；
  - 去重规则照 E1。
- **对照与基线**：
  - W1-0：E1 零样本，同一模型；
  - "自己的框"：refexp_box 的预测光栅化（`scripts/summarize_e1_posthoc.py` 的 own_box 读法）；
  - E1 结构强制多边形（`src/e1/poly_forced.py`，放宽子集检查）；
  - W1-F：自己的框直接交给 GrabCut，没有 LLM 的点、修正与核对。
  - 都在同一批句子上、同一个模型下跑。
- **输出**：`experiments/W1/<NNNN_slug>/<模型>/<任务>/<设置>/`，内含：
  - `predictions.jsonl`：逐条记录，含每轮的思考/作答原文、工具调用与结果摘要、最终答案、得分、compute；
  - `metrics.json`：含 bootstrap CI；
  - `config.yaml`：运行身份；
  - `README.md`。
  - 支持 `--resume`，按 id 续跑。

## 4. 排期（D103 §四）

| 时间 | 内容 |
|---|---|
| 10-20 → 10-23 | 三个开发任务；HPC 装 opencv-python-headless 与 scikit-image（RUN-0058，只用 CPU） |
| 10-23 → 10-25 | 07 做完 E5 事后探索后，在开发-调参集上调提示与工具，冻结回路协议；写 W1 预注册 |
| 10-25 → 10-27 | 06：W1-1 跑开发-检查集；07：补跑两条基线 |
| 10-27/28 | 去留判定（D103 §三，不得修改） |
