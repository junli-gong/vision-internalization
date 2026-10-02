# E2 开跑前的独立审查：探针读到的是不是它以为的那些东西（2026-10-02）

只读审查 agent 的报告，管理员整理存档（D53）。审查对象是 main `3350077`。核对脚本在 `docs/reviews/2026-10-02-e2-scripts/`：CPU fp32 小型随机模型、`HF_HUB_OFFLINE=1`，原样存档，路径可能要改。

**结论**：没有发现会让 E2 结论出错的问题。层号、图像区间、网格顺序、输入顺序、数值处理，在三个家族上都经源码与小模型实测核对通过。另有两条影响解读、三条与去重有关的小问题，以及网格测试的盲区。去重清单"缺整个训练集时静默放行"一条，建议开跑前修。

## 1. 层号：核对通过

- **源码依据**：
  - `transformers/utils/output_capturing.py:124`：第一个 decoder 层的输入 `args[0]` 记为第 0 项，也就是视觉特征并入之后的 `inputs_embeds`；
  - 之后每个 decoder 层的输出依次记一项；
  - `:305` 用 `last_hidden_state`（即 `norm(h)`）覆盖最后一项。
  - 三个家族都把 hidden_states 作为位置参数传给 decoder 层：Qwen3.5 `modeling_qwen3_5.py:1292`，Gemma4 `modeling_gemma4.py:1692`，Qwen3 `modeling_qwen3.py:413`。
  - Qwen3.5 的 GDN 层与全注意力层是同一个类（`Qwen3_5DecoderLayer`，`:865-903` 按 `block_type` 分支），混排不影响记录。
  - 三个适配器都断言元组长度为 `num_layers()+1`：`qwen35.py:1551`、`gemma4.py:1276`、`internvl35.py:1147`。
- **实测**（`check_layers.py`，三家族 × 两种顺序）：用钩子独立抓每层的输入、输出与 final norm 的输出。
  - L0 与第 0 层输入、`vision_features` 都逐位相同；
  - L1..L3 与第 i−1 层的原始输出逐位相同；
  - L4 等于 final norm 的输出，也等于 norm(第 4 层原始输出)，但不等于第 4 层原始输出；
  - 适配器路径与 HF 原生路径在全部层、全序列上的最大差是 0.0。
- **Gemma 的 PLE**：在每个 decoder 层 MLP 之后才加进残差（`modeling_gemma4.py:1431-1440`）。所以 L0 不含 PLE，L≥1 含。适配器在图像位置用 pad id 查 PLE 表，与原生路径相同。
- **Gemma 的滑窗**：窗口 512。E2 的序列约 278 个 token，没有图像 token 被窗口截断。真模型上这一点只按算术推出。
- **影响解读（I1）**：
  - 最后一个点是"第 N 层 + 最终 RMSNorm"。按通道标准化抵消不了逐 token 的 1/RMS 缩放，所以曲线最后一点与前面各点不是同一种量。
  - 建议另加 `L{N}_pre`：最后一个 decoder 层的原始输出。

## 2. 图像 token 的位置：核对通过

- **Qwen**：span 是 `<|image_pad|>` 的连续段，夹在 `<|vision_start|>` 与 `<|vision_end|>` 之间（`qwen35.py:909`）。实测 [4, 304)，300 个，网格 (15, 20)。
- **Gemma**：span 是 `<|image|>` 段，并与 `num_soft_tokens_per_image` 核对（`gemma4.py:620, 649`）。填充 patch 的位置是 (−1, −1)，由 pooler 的 mask 去掉。实测 266 个，网格 (14, 19)，有效 patch 2394 = 266 × 9。
- **InternVL**：span 是 `<IMG_CONTEXT>` 段（`internvl35.py:582`）。E1 口径是单 tile：处理器切出多于 1 个 tile 就报错（`:549`），探针方法对网格不是 (16, 16) 的一律拒绝（`:1046-1056`）。实测 256 个。

## 3. 网格排布：顺序核对通过；现有白块测试有盲区

- **源码依据**：
  - Qwen 的处理器按 (h/m, w/m, m, m) 展平，merger 把每 m² 个连续 patch 合成一个 token，`get_rope_index` 按行主序编号；
  - Gemma 的 pooler 核下标 = x//3 + (max_x//3)·(y//3)（`modeling_gemma4.py:651-653`）；
  - InternVL 的 `pixel_shuffle`（`modeling_internvl.py:662-696`）按 (y/2, x/2) 行主序。
- **元数据实测**（`order_source.py`，三种尺寸）：
  - Qwen：第 k 个图像 token 的 M-RoPE 恰为 (k//w, k%w)，每 4 个连续 patch 落在同一格；
  - Gemma：核下标 = (y//3)·w + x//3；
  - InternVL：第 k 个 token 的 4 个 patch 恰是第 (k//16, k%16) 格的 2×2 窗口。
- **几何**（`resize_geom.py`）：patch 拼回画布，与整图直接缩放相比，三家族最大差都是 0，即不裁、不补边。这是开发机的 PIL 后端结果，服务器上的 torchvision 后端未核实。
- **全格、逐子 patch 的白块扫描**（`sweep_grid.py`）：
  - Qwen：横、竖各 1200/1200 全部命中；
  - InternVL：1024/1024 全部命中；
  - Gemma：直接在 `vision_features` 上扫，有 82–180/2394 没有命中。诊断（`gemma_diag.py`、`gemma_noenc_sweep.py`）显示，原因是小模型随机编码器里的"枢纽" token，再被 `embed_vision` 的 RMSNorm 放大；旁路编码器后 0/2394 失配。所以不是排布错误。
- **对 DEV-0023 白块测试（`tests/test_e2_grid.py:89-107`）的判断**（`mut.py` 解析计算）：
  - **能查出**：h、w 写反；整体转置；翻转；错一位；token 级 2×2 块内转置；块优先的顺序。不过其中有的只靠 5 个格里的 1 个查出，很脆弱。
  - **查不出**：避开那 5 个格的局部换位；格内 patch 到 token 的归属错误。
  - 另外，灰底在 Gemma 上用 argmax 判据不可靠。
- **建议补测**：
  - (a) 用模型自己的元数据确定性地核对顺序；
  - (b) Qwen 与 InternVL 做全格逐子 patch 扫描；
  - (c) Gemma 在 pooler 输出上扫，或旁路编码器后扫。
- **附带观察**：Qwen 的像素预算 min = max = 300，smart_resize 实际给出 280–330 个 token（683×512 → 14×20，640×427 → 15×22）。这是 D28-3 口径的属性。

## 4. 因果注意力与输入顺序：核对通过

- 三家族的 `_content`：指令在前 → [指令, 图]，图在前 → [图, 指令]（`qwen35.py:1409-1415`、`gemma4.py:1116-1119`、`internvl35.py:986-989`）。E2 的稠密任务用图在前，指代分割用指令在前（`features.py:149, 190`）。
- **实测**：
  - 图在前、换一句等长提示：全部层在图像位置逐位相同；
  - 换一句更长的提示：Gemma 最大差 1.5e-5，是 CPU 上矩阵乘的分块随长度变化；
  - 指令在前、换一个指代句：L0 不变，L1 起每层都变。
- Gemma E2B 是纯因果，图像 token 之间也是因果。
- **影响解读（I2）**：
  - 三家族在 LLM 内部，图像 token 之间都按光栅顺序因果。第 (r, c) 格从第 1 层起只看得到光栅顺序在它之前的图像 token，所以"信息增加或丢失"可能随位置变化。
  - 建议按图像上半与下半、按光栅下标的分位，分开报指标。
  - 另外，指令在前不是 Gemma card 推荐的顺序（model_notes §6.6），解读指代分割时要注明。

## 5. 数值与设备：核对通过

- **冻结与缓存**：
  - 前向在 `no_grad` 下；`model.training` 为真就报错（`features.py:167-175`），隐藏状态带梯度也报错（`:222-223`）；
  - 适配器 attach 时调 `model.eval()`，前向带 `use_cache=False`；
  - 探针里没有 dropout；Qwen 显式传 position_ids。
- **数值**：
  - bf16 → fp32 在 `Normalizer.apply` 里转（`features.py:107`），统计量用 float64 累计；
  - 实测（`stats_check.py`）：均值误差/std 3.9e-13，方差相对误差 5.2e-14，`Normalizer` 与 fp64 的最大差 4.6e-5。

## 6. 数据泄漏

- **核对通过**：
  - 没有剔除清单就报错（`train_loaders.py:33-41`），E2 默认 `require_dedupe=True`；
  - 留出验证与标准化样本只从训练池取；
  - RefCOCO 按 D11-1 剔除；
  - NYU 按 D11-2 做场景过滤，test 场景表钉了 sha256，之后再由感知哈希兜底。
- **小问题 A（建议开跑前修）**：`load_exclusions(path).get(set_name, {})`（`train_loaders.py:41`）。清单在、但缺这个训练集的键时，返回空剔除，不报错。仓库自己的 `--tiny` fixture 就是这样。
- **小问题 B**：剔除清单不记与哪些评测集比过，所以 E2 无法核实自己的评测集都在比较范围内。
- **小问题 C**：`load_nyu_depth_train` 不复查场景过滤。
- **流程提醒**：感知哈希的阈值还是暂定值（D49-3），正式的剔除清单要在阈值定下之后生成。

## 未核实（没有 GPU，也没有真权重）

- 真模型 bf16 GPU 上：`vision_features` 与 L0 是否逐位相同（E2 只在加 `--reference post_proj` 时核对）；fla GDN 内核的数值；torchvision 后端的缩放几何。
- 真模型上 Gemma 的序列长度小于 512，只按算术推出。
- Gemma E4B 与 Qwen 2B–9B 的 config 本地没有缓存。
- A800 上实际生成的剔除清单的内容。
