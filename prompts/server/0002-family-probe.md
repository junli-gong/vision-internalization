[RUN-TASK 0002] Phase 0：Gemma 4 E2B-it 与 InternVL3.5-2B-HF 的事实探针（照 RUN-0001 §5 的格式）

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D6（磁盘红线）、D9（变体与 revision）
- 你在 RUN-0001 写的 `docs/model_notes.md` §5 和 `docs/env_report.md`：本任务对另两个家族做同样的事，表格格式尽量照抄，方便三家族并排比较。

不写实验逻辑代码。探针脚本放 `$PROJ/scripts/`，副本进 `experiments/P0/0002_family_probe/probe_scripts/`。

## 1. 拉取与校验

`git pull --ff-only`，确认 `git log --oneline -1 -- prompts/server/0002-family-probe.md` 指向的提交在 HEAD 历史里。本任务不跑仓库代码，不需要测试。

## 2. 磁盘与内存盘（D6、D9）

- 规则照旧：根盘剩余任何时刻低于 3 GB 立即停止；根盘新增占用不超过 1 GB；内存盘不超过 25 GB。
- 先删 `/dev/shm/visint/hf-hub` 里的 Qwen3.5-9B（只删 `models--Qwen--Qwen3.5-9B` 这一个目录），为两个新模型腾地方；Qwen3.5-0.8B 保留。删之前和删之后的占用都记进 report。
- 下载时固定 revision，取 `docs/model_notes.md` §2 登记的 sha：
  - `google/gemma-4-E2B-it` @ `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
  - `OpenGVLab/InternVL3_5-2B-HF` @ `3f301ffcf3dcbb47893afae6650ea3e78d96fb6d`

## 3. 事实探针（两个模型各做一遍；生成一律 greedy，显式设置思考开关）

**3.1 结构**
- LLM 层数与每层类型：Gemma 的 sliding/full 分布，以及 `num_kv_shared_layers` 的含义。
- hidden size，视觉编码器（patch、pool 或 downsample），每个 LLM 视觉 token 对应多少像素。
- 词表大小与嵌入行数，有没有预留空行或未使用的特殊 token。
- 新增一个 token（`<g0>`）：是不是单个 token，嵌入怎么扩展，`resize_token_embeddings` 的行为（会不会像 Qwen 那样缩行）。
- Gemma 的 per-layer embedding（`hidden_size_per_layer_input`）会不会影响"按层取隐藏状态"：`output_hidden_states` 返回几个张量、形状各是什么。

**3.2 聊天模板与思考开关**
- 开关怎么传，关闭时带图消息渲染出的完整文本（原样贴）。
- `generation_config` 的默认值；model card 推荐的采样参数，原文引用并附链接。
- 生成在哪个 token 停（eos 是哪些 id）。

**3.3 图像 token 化**
- 怎么固定分辨率和视觉 token 数，给出参数名：
  - Gemma：`max_soft_tokens` 的可选档位；
  - InternVL：`max_patches` / `crop_to_patches` 与 thumbnail。
- 448×448、640×480、1024×768 三种输入各产生多少视觉 token，网格 (h, w) 是什么。
- 两张图同时输入时的排布。

**3.4 位置编码**：图像 token 和文本 token 的位置怎么编（一维，还是有二维成分），贴一个"指令 + 图 + 文本"最小例子的 position ids 摘要。

**3.5 原生坐标格式**
- 文档怎么说：原文加链接。特别注意轴序（x 先还是 y 先）和量程（像素、0–1000 还是 0–1）。
- 用 RUN-0001 那 2 张 COCO val2017 图、同样的目标，各做一次零样本定位，贴原始输出。提示词：Gemma 用 model card 或官方示例里的定位写法，InternVL 用它 card 里的 grounding 写法，都找不到就用 RUN-0001 的那句。写明用的是哪句、出处在哪。不要据输出调提示词。

**3.6 任务调制仪器（D1）**：同 RUN-0001 §5.7。一张图，两条 token 长度相同的指令，batch=1，取图像 token 位置的各层隐藏状态：
- "图在前"应当各层逐位相同；
- "指令在前"应当第 0 层相同、之后各层不同。

贴每层的 torch.equal、max|Δ|、cos。Gemma 的 model card 要求图在文字之前，两种顺序照样都跑，并记下 card 原文。"图在前"如果不是逐位相同，标"仪器异常"，把可能相关的结构事实（sliding window、KV 共享）一并记下，不要解读。

**3.7 显存与吞吐**：只测 bf16：
- 推理：约 1K 视觉 token 或该家族最接近的档位，batch 1 和 16；
- LoRA：r=32，只作用于语言模型的线性层，开梯度检查点，batch 8，单步前向加反向；
- HF generate 吞吐：batch 16，`max_new_tokens=64`，关闭思考。

## 4. 产出

- `docs/model_notes.md`：新增"§6 Gemma 4 E2B-it 事实探针"和"§7 InternVL3.5-2B-HF 事实探针"。
- `docs/env_report.md`：新增一节"§8 另两个家族的显存与吞吐"。
- `experiments/P0/0002_family_probe/`：`probe_scripts/`、`raw/`（单个文件不超过 5 MB）、`README.md`。
- `reports/0002-family-probe.md`：按 report 模板，含磁盘与内存盘收支。

## 5. 回传

分两批：先推 Gemma（§3 全部），再推 InternVL 和 report。

```bash
git add <逐个路径> && git commit -m "[results] 0002: <本批摘要>" && git push
```

完成后按 STANDBY.md，把 `0002-family-probe.md` 追加进完成清单，重新进入值守。
