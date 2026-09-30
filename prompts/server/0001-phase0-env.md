[RUN-TASK 0001] Phase 0：环境核查、Qwen3.5 事实探针、模型登记、数据源体积与 schema（磁盘受限版）

## 0. 先读

- `CLAUDE.md`
- `.claude/skills/exp-triad-flow/SKILL.md` 与 `references/server.md`（你的纪律）
- `docs/RESEARCH_BRIEF.md` 的 §4（硬约束）、§5、§6、§7、§8.1、§12
- `DECISIONS.md` 的 D1、D3、D6

本次不写任何实验逻辑代码。探针脚本写在 `$PROJ/scripts/`，跑完把副本放进仓库当证据（见 §7）。

## 1. 拉取与校验

```bash
cd /home/ubuntu/workspace/wzg/gjl/vision-internalization/repo
git pull
git log --oneline | head -3   # 应包含 1fdeb1d（bootstrap），以及加入本任务书的那条 [mgr] commit；不含就停下报告
```

本次没有测试可跑（代码骨架由 DEV-0001 在开发机上并行开发）。

## 2. 磁盘红线（D6，最高优先）

- 整机只有根盘 `/` 这一块。2026-09-29 剩 6.1 GB，其他租户也在用。
- **任何时刻 `df /` 剩余低于 3 GB，立即停止当前步骤并报告。**
- 本次在根盘上的新增占用上限 3 GB（venv 加小文件）。
- 模型权重一律放内存盘 `/dev/shm/visint/hf-hub`，上限 25 GB。
- 每次下载前先探体积（HF API 的文件大小，或 HTTP HEAD），记进 report 的磁盘收支表：每一步前后的 `df /` 与 `du -sh /dev/shm/visint`。
- 不删除、不移动任何非本项目文件，包括 `$ROOT/mmvis`、`$ROOT/datasets/rgbt`、`$ROOT/.hf` 里的旧模型。回收与否等 DECISIONS_NEEDED N1。

## 3. 环境（结果写进 `docs/env_report.md`）

所有脚本自己 export：

```bash
export ROOT=/home/ubuntu/workspace/wzg/gjl PROJ=$ROOT/vision-internalization
export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub TMPDIR=$PROJ/tmp PIP_NO_CACHE_DIR=1
```

1. 建 venv：`$ROOT/env/bin/python -m venv --system-site-packages $PROJ/venv`（不改 `$ROOT/env`）。
2. 安装：
   - transformers：先装最新 release；不支持 Qwen3.5 再装 main，并记录 commit。
   - 其余：accelerate、peft、flash-linear-attention、einops、huggingface_hub、pyyaml、pytest、shapely、pycocotools、scipy、pillow。已有的跳过。
   - 若官方示例需要 `qwen-vl-utils` 一类的辅助包，一并装并记录。
   - 每装一批看一次 `df /`。
3. 记录：
   - 驱动、CUDA、torch 版本，以及 `torch.cuda.get_device_capability()`（期望 (10, 0)）。
   - 各包版本，`nvidia-smi` 头部。
   - `/` 与 `/dev/shm` 的剩余。
4. **vLLM 与 SGLang 只做 dry-run，不安装。** 对主分支或 nightly 各跑一次 `pip install --dry-run`，记录：
   - 它会装哪些包；
   - 会不会另装一个与 2.13.0 不同的 torch；
   - 下载总体积。

   磁盘问题解决前不装。

## 4. 模型登记（不下载权重；写进 `docs/model_notes.md`）

用 `HfApi().model_info(repo, files_metadata=True)` 核实下列每个候选仓库，记录：

- 仓库 ID、当前 revision（commit sha）、license；
- 是否 gated，以及我们的 token 能否访问；
- safetensors 总大小；
- config 里的层数、hidden size、注意力布局。

候选仓库：

- **Qwen3.5 后训练版：** 0.8B、2B、4B、9B。已知有 `Qwen/Qwen3.5-4B`、`Qwen/Qwen3.5-9B`，其余在 Qwen 组织下检索确认，不要猜名字。
- **Gemma 4：** E2B、E4B 的指令版。
- **InternVL3.5：** 2B、4B、8B。HF 格式优先，记录各有几种变体。

找不到或有歧义的写明，不替我们选。

## 5. Qwen3.5 事实探针（GPU；只下载 0.8B 和 9B 到内存盘）

- 生成一律 greedy，显式设置思考开关。
- 原始输出全部落盘到 `$PROJ/logs/0001/`，报告里原样贴关键部分。
- 9B 下载或运行出错时，只做 0.8B，写明原因。

### 5.1 结构

- 层数，每层类型（全注意力还是 Gated DeltaNet），hidden size，头数。
- 视觉编码器的 patch size 和空间合并倍数。
- 词表大小与嵌入行数：有没有预留的空行，有几行。
- tokenizer 里未使用的特殊 token：名字、id、个数。
- 新增一个 token（例如 `<g0>`）后：它是否被当作单个 token；嵌入怎么扩展，是复用预留行还是 resize。

### 5.2 聊天模板与思考开关

- `enable_thinking` 的传法：`apply_chat_template` 的参数，还是 `chat_template_kwargs`。
- 关闭思考时，模板实际渲染出的完整文本，原样贴出。
- `generation_config` 的默认采样参数。
- 模型卡上两种模式各自的推荐采样参数：原样引用，附链接。

### 5.3 图像 token 化

- 用处理器参数固定分辨率的方法：给出参数名。
- 448×448、640×480、1024×768 三种输入各产生多少视觉 token，网格 (h, w) 是多少，每个 token 对应多少像素。
- 两张图同时输入时，token 如何排布。

### 5.4 位置编码

- 图像与文本 token 的 `position_ids` 形状（M-RoPE 各维）。
- 图像之后的文本，位置索引怎么接上。
- 贴一个"指令 + 图 + 文本"最小例子的 `position_ids` 摘要。

### 5.5 原生坐标格式

- 官方文档、模型卡、示例对定位输出坐标的说法：绝对像素还是 0–1000。原样引用，附链接。
- 用 2 张 COCO val2017 图（按 URL 单张下载），各做一次"给出某物体的框"零样本，贴原始输出。不要据此调提示词。

### 5.6 Gated DeltaNet 数值

1. 同一组随机小输入（fp32、bf16 各一组；序列长度至少覆盖 2 个 chunk），分别过三种实现，贴两两之间的最大绝对误差与相对误差：
   - fla 的 chunk 内核；
   - fla 的朴素或递归参考实现；
   - transformers 模型文件里自带的 torch 实现（有 chunk 和 recurrent 两种就都测）。
2. 0.8B 整模型、同一输入：fla 路径与 torch 回退路径的 logits 最大误差。
3. transformers 实际走的是哪条路径（fla、causal_conv1d 还是 torch），有没有警告。

### 5.7 隐藏状态与任务调制（仪器，D1）

取一张图和两条 **token 长度相同** 的不同指令，分别以 batch=1、同一 dtype、同一内核各跑一次，取图像 token 位置的各层隐藏状态：

- **图在前：** 两条指令下各层应**逐位相同**（`torch.equal`）。贴每层的 True/False。
- **指令在前：** 第 0 层（embedding 输出）应逐位相同，之后各层应不同。贴每层的 max|Δ| 与余弦相似度。

"图在前"若不是逐位相同：贴出每层的 max|Δ|，标为"仪器异常"，继续做其他步骤，不要解读。

### 5.8 显存与吞吐实测

结果写进 `env_report.md` 的表，替换简报 v0.1 §5.4 的估算。0.8B 与 9B 各测：

- **推理：** bf16，约 1K 视觉 token，batch 1 和 16，峰值显存。
- **在线探针式前向：** `output_hidden_states=True` 取全部层，batch 8，峰值显存。
- **LoRA：** r=32，作用于全部线性层，开梯度检查点，batch 8，单步前向加反向的峰值显存。只测，不训练，不保存。
- **HF generate 吞吐：** batch 16，`max_new_tokens=64`，关闭思考，样本/秒。

## 6. 数据源（不落地大数据；写进 `docs/data_sources.md`）

对简报 §7.1–7.3 的每个数据集，记录：

- 候选来源（官方 URL 或 HF 仓库）、许可、划分与条数；
- 总体积（用 HEAD 或 API 查）；
- 我们实际需要的子集体积。例如 RefCOCO 系列只需要它引用到的 COCO train2014 图：算出去重后的图数与体积，并确认能否按单图 URL 下载。
- 真实 schema：
  - HF 数据集：用 datasets-server 的 first-rows API 贴字段名和类型，不占磁盘；
  - 非 HF 来源：贴官方格式说明的链接。

特别核实并记录（只记事实，不下结论）：

- **CV-Bench：** 图像来自哪些源数据集的哪个划分（ADE20K、COCO、Omni3D）；Omni3D 部分是否包含 SUN RGB-D，进而包含 NYUv2 的图。这是为去重做准备。
- **NYUv2：**
  - 官方 795/654 划分。`$ROOT/datasets/nyuv2/splits.mat` 已在盘上（2.6 KB），可读不可改。
  - 常用的约 2.4 万帧训练集：有哪些可得来源，各多大。
  - 法向 GT 的来源。
- **BLINK 相对深度：** val 和 test 是否都公开答案。
- **CountBench：** 图像是 URL 还是打包提供；失效链接比例，抽 50 条做 HEAD 探测即可。
- **只有测试划分的 benchmark：** 列出清单。
- **Hypersim（可选）：** 许可与子集体积。

为看 schema，允许下载单个小于 50 MB 的标注样本文件，总量小于 200 MB。放在 `$PROJ/tmp`，report 里记录，看完删除。

## 7. 产出

- `docs/env_report.md`、`docs/model_notes.md`、`docs/data_sources.md`：事实为主，不解读。
- `experiments/P0/0001_env_probe/`：
  - `probe_scripts/`：脚本副本；
  - `raw/`：原始输出，json 或 txt，单个文件不超过 5 MB；
  - `README.md`：说明每个文件是什么。
- `reports/0001-phase0-env.md`：按 `.claude/skills/exp-triad-flow/assets/report.template.md` 写，第 6 节磁盘收支必须有。

## 8. 回传

分三批 push，每批完成就推，管理员会提前开始读：

1. §3 + §4：`env_report.md`、`model_notes.md`
2. §5：探针结果和 `experiments/P0/0001_env_probe/`
3. §6 + report

```bash
git add <逐个路径> && git commit -m "[results] 0001: <本批摘要>" && git push
```

- 环境问题可以修，全部记录。
- 遇到与简报硬约束相关的疑问，停下写进 report。
- 内存盘里的权重跑完保留，后续 RUN 复用。report 里写明占用。
