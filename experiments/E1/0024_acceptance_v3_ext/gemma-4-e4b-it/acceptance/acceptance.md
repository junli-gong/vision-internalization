# 批处理验收 v3 · gemma-4-e4b-it

- 规则（configs/e1_prompts.json 的 acceptance，v3）：开发子集上，(i) 同一批配置（batch_protocol）跑两遍（A1、A2），每个任务的 predictions.jsonl 逐字节相同；(ii) 换一种批组成（B，见 composition_b）再跑一遍，按组比较 A1 与 B 的标题指标（严格读数）：两遍逐样本配对、bootstrap 的 95% CI 落在 [−δ, +δ] 以内（δ 按组的 metric_class 取 delta）。所有任务过 (i)、所有能判的组过 (ii) 才算通过。A1 与 B 之间的指标差及其 CI 记作数值噪声底，写进结果；E3 里小于它的效应不做解读。不再以 batch-1 为参照（D36-1）；三种一致口径（逐位、IoU ≥ 0.9、按判定结果）在 A1 与 B 之间照算，只作参考。
- 本次 A1、A2：batch_size = 64，left 填充，顺序 selection，same_letters；B：batch_size = 48。是协议的冻结值（A 64、B 48）
- composition_b 的规则：批大小 48，其余同 batch_protocol（顺序不变、左填充、受限字母读数按字母集合分批）。批大小与 64 不同，除了被字母集合变化或末尾截短的批，B 里没有哪一批与 A 的组成相同：每条样本的同批邻居与填充长度都变了。比 64 小，显存不会比正式运行更紧。D43 随 A 由 16/12 改为 64/48。
- δ：{'accuracy': 0.02, 'iou': 0.02}；组与合并规则见下表与配置的 groups；组的 take 是整数、且某个任务的开发样本多于它时，按 sha256(salt|<take_salt_purpose>:<组名>|样本 id) 升序取前 take 个（src/data/subsets.seeded_sample），再按原样本顺序排列；take = all 时全取。--limit（只用于调试与 --tiny）在 take 之前截断。
- {'model': 'gemma-4-e4b-it', 'revision': 'ee0ef6023621cff504d758262d4e04895a5af4a2', 'policy': 'default', 'device': 'cuda', 'dtype': 'bf16', 'tiny': False, 'prompt_config_sha256': '5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158', 'batch_size_a': 64, 'batch_size_b': 48, 'frozen': True, 'limit': None, 'count_questions': 'countbenchqa', 'gpu': 'NVIDIA B200'}

## 判定

- all_pass：**否**；(i) 重跑逐位相同：是（不同的任务：—）；(ii) 不通过的组：polygon；不判的组：—；n 低于 n_min 的组：—

## 数值噪声底（A1 与 B 之间，按组）

| 组 | 任务 | 统计量 | n | n_min | A1 | B | 差（A1 − B） | 配对差 95% CI | δ | CI ⊂ ±δ |
|---|---|---|---|---|---|---|---|---|---|---|
| short_text | refexp_box、count | accuracy_pooled | 649 | 300 | 0.2296 | 0.2280 | 0.0015 | [-0.0031, 0.0062] | 0.0200 | 是 |
| letter | rel_depth_nyu、rel_depth_blink、spatial_cvbench、spatial_embspatial | accuracy_pooled | 362 | 300 | 0.5387 | 0.5387 | 0.0000 | [0.0000, 0.0000] | 0.0200 | 是 |
| polygon | refexp_poly | cIoU | 500 | 500 | 0.0691 | 0.0598 | 0.0093 | [-0.0024, 0.0246] | 0.0200 | 否 |
| grid | grid16 | gIoU | 500 | 500 | 0.0159 | 0.0158 | 0.0001 | [-0.0013, 0.0015] | 0.0200 | 是 |
| detection | ovd | F1@0.5 | 500 | — | 0.4941 | 0.4987 | -0.0046 | [-0.0104, 0.0009] | 0.0200 | 是 |

## 逐任务（(i) 的判定；A1 与 B 之间的一致率与指标差只作参考）

| 任务 | 组 | n | (i) A1 = A2 逐位 | 不同的条数 | (a) 逐位 一致率（A1 vs B） | (b) IoU ≥ 0.9 一致率（A1 vs B） | (c) 按判定结果 一致率（A1 vs B） | raw_text 逐字相同（A1 vs B） | 指标 | A1 | B | 差 | 配对差 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refexp_box | short_text | 600 | 是 | 0 | 0.8850 | 0.9933 | 0.9950 | 0.6933 | acc@0.5 | 0.2017 | 0.2000 | 0.0017 | [-0.0033, 0.0067] |
| refexp_poly | polygon | 500 | 是 | 0 | 0.9340 | 0.9340 | 0.9340 | 0.1860 | cIoU | 0.0691 | 0.0598 | 0.0093 | [-0.0024, 0.0246] |
| grid16 | grid | 500 | 是 | 0 | 0.9860 | 0.9860 | 0.9860 | 0.9860 | gIoU | 0.0159 | 0.0158 | 0.0001 | [-0.0013, 0.0015] |
| count | short_text | 49 | 是 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5714 | 0.5714 | 0.0000 | [0.0000, 0.0000] |
| ovd | detection | 500 | 是 | 0 | 0.2360 | 0.8320 | 0.9000 | 0.2360 | F1@0.5 | 0.4941 | 0.4987 | -0.0046 | [-0.0104, 0.0009] |
| rel_depth_nyu | letter | 250 | 是 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5520 | 0.5520 | 0.0000 | [0.0000, 0.0000] |
| rel_depth_blink | letter | 12 | 是 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.4167 | 0.4167 | 0.0000 | [0.0000, 0.0000] |
| spatial_cvbench | letter | 50 | 是 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5200 | 0.5200 | 0.0000 | [0.0000, 0.0000] |
| spatial_embspatial | letter | 50 | 是 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5200 | 0.5200 | 0.0000 | [0.0000, 0.0000] |
