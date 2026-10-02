# RUN-0020 — E1 批处理验收 v3（冻结批配置 64，B = 48；先跑 4 个模型）

服务器 CC，b200，2026-10-02 08:59–10:53 UTC，值守中接单。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- **总判定：** 4 个模型里 3 个 `all_pass=True`：qwen3.5-0.8b、gemma-4-e2b-it、internvl3.5-2b-hf。qwen3.5-9b 是 `all_pass=False`，`groups_failed=['detection']`。
- **(i) 逐位重跑：** 4 个模型 × 9 个任务，A1 与 A2 的 `predictions.jsonl` 全部逐字节相同，`rerun_all_identical=True` × 4。
- **(ii) 噪声底：** 20 个"模型 × 组"里 19 个的 95% CI 在 ±0.02 以内。唯一不在的是 qwen3.5-9b 的 detection：
  - F1@0.5：A1 0.5859，B 0.5742；
  - 差 +0.0117，CI [−0.0026, +0.0266]，上界超过了 +0.02。
- **没有扩到全部 8 个模型。** 任务书 §3 只给了这 4 个模型的命令；"不达标再扩到全部 8 个"写在"为什么做"里，引的是 D38-5。我没有自行扩展，见 §4。
- **internvl3.5-2b-hf 的检测：** A1 的 500 条里 491 条 `no_box`、9 条截断，严格 F1 为 0；回答是文字描述，没有框。B 也是 0，所以这一组的差与 CI 都是 0。
- **Gemma E2B 的定位拒答：** person 283 条里拒答 176 条（0.622），非 person 317 条里拒答 106 条（0.334）。
- **用时与磁盘：** 累计 6,803 s（1.89 h），没有触到 7 h 上限。根盘 38.182 → 38.081 GB，`$PROJ` 4.556 → 4.634 GB，内存盘峰值 22.525 GB。

## 1. 前置校验

- **HEAD：** `git pull --ff-only` 之后 HEAD 是 `90ab6b8`。`2257e08`（D43）与本任务书所在的 `97d12d1` 都在 HEAD 的历史里。
- **配置：** `configs/e1_prompts.json` 的 sha256 是 `5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158`，与任务书相同；各 acceptance.json 的 header 也是这个值。
- **环境：**
  - `pip install --dry-run -r requirements.txt` 没有任何 "Would install"。
  - 环境变量按任务书 §2 设置。
  - `coco_train2014_e1dev` 与 `countbenchqa_questions` 都已落地，没有重落。
- **运行用的代码：** 全部在固定于 `90ab6b8` 的 worktree 里运行，跑完已删除。
- **改动范围：**
  - 没有修改 `src/`、`scripts/`、`tests/`、`configs/`，也没有安装 Python 包。
  - 新写的文件只在 `probe_scripts/` 下：`run_accept_v3.sh`、`p25_accept_v3_summary.py`。
- **HOLD：** 运行期间 origin 上没有 `HOLD` 文件。
- **只用了开发子集。**

## 2. 执行记录

| # | 模型 | 开始时累计 s | 权重核对 | run_e1 墙钟（`time` real） | rc | all_pass | nvidia-smi 峰值 MiB | 删除释放 |
|---|---|---|---|---|---|---|---|---|
| 1 | qwen3.5-0.8b | 0 | 13/13 | 27m38.9s | 0 | True | 9596 | 保留 |
| 2 | gemma-4-e2b-it | 1660 | 9/9 | 17m00.1s | 0 | True | 16588 | 10.279 GB |
| 3 | internvl3.5-2b-hf | 2705 | 18/18 | 27m25.3s | 0 | True | 26964 | 4.715 GB |
| 4 | qwen3.5-9b | 4365 | 16/16 | 39m55.1s | 0 | False | 34550 | 19.317 GB |

- 结束于 10:53:05，累计 6,803 s。
- **GPU：** NVIDIA B200，bf16。nvidia-smi 每 5 s 采样一次。
- **报错：** 4 次 run_e1 都没有 Traceback。

## 3. 结果表（原样）

由 `probe_scripts/p25_accept_v3_summary.py` 从各 `acceptance.json` 与 A1/A2/B 三遍的 `metrics.json`、`compute.json` 生成，与 `accept_v3_summary.md` 相同。

#### 总判定（acceptance.json）

| 模型 | all_pass | rerun_all_identical | rerun_not_identical | groups_failed | groups_not_evaluated | groups_below_n_min | 批大小 A / B | frozen |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | 是 | 是 | [] | [] | [] | [] | 64 / 48 | 是 |
| gemma-4-e2b-it | 是 | 是 | [] | [] | [] | [] | 64 / 48 | 是 |
| internvl3.5-2b-hf | 是 | 是 | [] | [] | [] | [] | 64 / 48 | 是 |
| qwen3.5-9b | 否 | 是 | [] | ['detection'] | [] | [] | 64 / 48 | 是 |

#### (i) 逐位重跑：A1 与 A2 的 predictions.jsonl（tasks[].rerun）

| 模型 | 任务 | n | 文件逐字节相同 | 逐条不同的条数 | raw_text 不同的条数 | A1 的 sha256（前 16 位） |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 600 | 是 | 0 | 0 | a6a862eaacb75cef |
| qwen3.5-0.8b | refexp_poly | 500 | 是 | 0 | 0 | e164caf5f9c0134d |
| qwen3.5-0.8b | grid16 | 500 | 是 | 0 | 0 | 1033fecec42496ac |
| qwen3.5-0.8b | count | 49 | 是 | 0 | 0 | cc3a05e6a3066c46 |
| qwen3.5-0.8b | ovd | 500 | 是 | 0 | 0 | f8ad81907dbe714d |
| qwen3.5-0.8b | rel_depth_nyu | 250 | 是 | 0 | 0 | a1617fc498ad90e5 |
| qwen3.5-0.8b | rel_depth_blink | 12 | 是 | 0 | 0 | 0008ceed1552c301 |
| qwen3.5-0.8b | spatial_cvbench | 50 | 是 | 0 | 0 | 5fe8d98dc7c7b790 |
| qwen3.5-0.8b | spatial_embspatial | 50 | 是 | 0 | 0 | a1061cb3b577aee5 |
| gemma-4-e2b-it | refexp_box | 600 | 是 | 0 | 0 | 8db936fb037e409b |
| gemma-4-e2b-it | refexp_poly | 500 | 是 | 0 | 0 | 65a9d658143747c2 |
| gemma-4-e2b-it | grid16 | 500 | 是 | 0 | 0 | 852414e786973d66 |
| gemma-4-e2b-it | count | 49 | 是 | 0 | 0 | 047867798563754d |
| gemma-4-e2b-it | ovd | 500 | 是 | 0 | 0 | 086779e5d809b774 |
| gemma-4-e2b-it | rel_depth_nyu | 250 | 是 | 0 | 0 | b2e81850ad80d2d0 |
| gemma-4-e2b-it | rel_depth_blink | 12 | 是 | 0 | 0 | ee8cff49556cf53a |
| gemma-4-e2b-it | spatial_cvbench | 50 | 是 | 0 | 0 | 1b3a3ef24f22feeb |
| gemma-4-e2b-it | spatial_embspatial | 50 | 是 | 0 | 0 | a5b6405a72a6eb28 |
| internvl3.5-2b-hf | refexp_box | 600 | 是 | 0 | 0 | f74c365e299dbe94 |
| internvl3.5-2b-hf | refexp_poly | 500 | 是 | 0 | 0 | 7009069c74f36f98 |
| internvl3.5-2b-hf | grid16 | 500 | 是 | 0 | 0 | 7d7d116597ea99c8 |
| internvl3.5-2b-hf | count | 49 | 是 | 0 | 0 | f37778db171fb252 |
| internvl3.5-2b-hf | ovd | 500 | 是 | 0 | 0 | f27b9c5e62d41a47 |
| internvl3.5-2b-hf | rel_depth_nyu | 250 | 是 | 0 | 0 | 15b05ce5b05932ff |
| internvl3.5-2b-hf | rel_depth_blink | 12 | 是 | 0 | 0 | 8ed132bcaab4f9f4 |
| internvl3.5-2b-hf | spatial_cvbench | 50 | 是 | 0 | 0 | 82425dc03756b1da |
| internvl3.5-2b-hf | spatial_embspatial | 50 | 是 | 0 | 0 | 3e5c968c898c0a17 |
| qwen3.5-9b | refexp_box | 600 | 是 | 0 | 0 | 0b6148aceea921e8 |
| qwen3.5-9b | refexp_poly | 500 | 是 | 0 | 0 | bd651b7e427b3e50 |
| qwen3.5-9b | grid16 | 500 | 是 | 0 | 0 | 0192aa1e5f120b5d |
| qwen3.5-9b | count | 49 | 是 | 0 | 0 | ca69380685ad9e5f |
| qwen3.5-9b | ovd | 500 | 是 | 0 | 0 | 064af16ebf8d5497 |
| qwen3.5-9b | rel_depth_nyu | 250 | 是 | 0 | 0 | 0feacec94a7803ae |
| qwen3.5-9b | rel_depth_blink | 12 | 是 | 0 | 0 | aa1ae9cf3c9c0401 |
| qwen3.5-9b | spatial_cvbench | 50 | 是 | 0 | 0 | 7183317e4ca1a672 |
| qwen3.5-9b | spatial_embspatial | 50 | 是 | 0 | 0 | 876f095b1fc63939 |

#### (ii) 噪声底：A1 与 B 按组比较（groups[]；严格读数，配对 bootstrap 95% CI）

| 模型 | 组 | 任务 | 指标 | n（n_min） | A1 | B | 差（A1−B） | 95% CI | δ | CI 在 ±δ 内 | 通过 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | short_text | refexp_box, count | accuracy_pooled | 649（300） | 0.8074 | 0.8089 | -0.0015 | [-0.0046, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-0.8b | letter | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | accuracy_pooled | 362（300） | 0.6298 | 0.6298 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-0.8b | polygon | refexp_poly | cIoU | 500（500） | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-0.8b | grid | grid16 | gIoU | 500（500） | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-0.8b | detection | ovd | F1@0.5 | 500（None） | 0.2573 | 0.2539 | 0.0034 | [-0.0031, 0.0108] | 0.02 | 是 | 是 |
| gemma-4-e2b-it | short_text | refexp_box, count | accuracy_pooled | 649（300） | 0.2080 | 0.2065 | 0.0015 | [0.0000, 0.0046] | 0.02 | 是 | 是 |
| gemma-4-e2b-it | letter | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | accuracy_pooled | 362（300） | 0.4448 | 0.4448 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| gemma-4-e2b-it | polygon | refexp_poly | cIoU | 500（500） | 0.2478 | 0.2443 | 0.0035 | [-0.0057, 0.0127] | 0.02 | 是 | 是 |
| gemma-4-e2b-it | grid | grid16 | gIoU | 500（500） | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| gemma-4-e2b-it | detection | ovd | F1@0.5 | 500（None） | 0.2796 | 0.2823 | -0.0027 | [-0.0099, 0.0034] | 0.02 | 是 | 是 |
| internvl3.5-2b-hf | short_text | refexp_box, count | accuracy_pooled | 649（300） | 0.7180 | 0.7165 | 0.0015 | [-0.0031, 0.0062] | 0.02 | 是 | 是 |
| internvl3.5-2b-hf | letter | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | accuracy_pooled | 362（300） | 0.5359 | 0.5359 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| internvl3.5-2b-hf | polygon | refexp_poly | cIoU | 500（500） | 0.0890 | 0.0947 | -0.0056 | [-0.0141, 0.0015] | 0.02 | 是 | 是 |
| internvl3.5-2b-hf | grid | grid16 | gIoU | 500（500） | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| internvl3.5-2b-hf | detection | ovd | F1@0.5 | 500（None） | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-9b | short_text | refexp_box, count | accuracy_pooled | 649（300） | 0.8891 | 0.8891 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-9b | letter | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | accuracy_pooled | 362（300） | 0.8232 | 0.8232 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-9b | polygon | refexp_poly | cIoU | 500（500） | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-9b | grid | grid16 | gIoU | 500（500） | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-9b | detection | ovd | F1@0.5 | 500（None） | 0.5859 | 0.5742 | 0.0117 | [-0.0026, 0.0266] | 0.02 | 否 | 否 |

#### 三列（A1 的 metrics.json 的 three_columns：严格读数 / 有效率 / 条件读数）

| 模型 | 任务 | 指标（范围） | 严格 | 严格 95% CI | 有效 n_valid/n | 条件 | 无效：拒答 / 其他失败 |
|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | acc@0.5（all） | 0.8083 | [0.7767, 0.8400] | 593/600 | 0.8179 | 0 / 7 |
| qwen3.5-0.8b | refexp_poly | cIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-0.8b | refexp_poly | gIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-0.8b | grid16 | gIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-0.8b | grid16 | cIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-0.8b | count | accuracy（all） | 0.7959 | [0.6735, 0.8980] | 49/49 | 0.7959 | 0 / 0 |
| qwen3.5-0.8b | ovd | F1@0.5（ovd_dev/refcoco_dev_images） | 0.2573 | [0.2257, 0.2871] | 233/500 | 0.6037 | 0 / 267 |
| qwen3.5-0.8b | rel_depth_nyu | accuracy（all） | 0.6760 | [0.6160, 0.7280] | 250/250 | 0.6760 | 0 / 0 |
| qwen3.5-0.8b | rel_depth_blink | accuracy（all） | 0.5833 | [0.3333, 0.8333] | 12/12 | 0.5833 | 0 / 0 |
| qwen3.5-0.8b | spatial_cvbench | accuracy（all） | 0.4800 | [0.3400, 0.6200] | 50/50 | 0.4800 | 0 / 0 |
| qwen3.5-0.8b | spatial_embspatial | accuracy（all） | 0.5600 | [0.4200, 0.7000] | 50/50 | 0.5600 | 0 / 0 |
| gemma-4-e2b-it | refexp_box | acc@0.5（all） | 0.1817 | [0.1533, 0.2100] | 151/600 | 0.7219 | 282 / 167 |
| gemma-4-e2b-it | refexp_poly | cIoU（all） | 0.2478 | [0.2224, 0.2706] | 494/500 | 0.2497 | 0 / 6 |
| gemma-4-e2b-it | refexp_poly | gIoU（all） | 0.2320 | [0.2123, 0.2502] | 494/500 | 0.2348 | 0 / 6 |
| gemma-4-e2b-it | grid16 | gIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| gemma-4-e2b-it | grid16 | cIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| gemma-4-e2b-it | count | accuracy（all） | 0.5306 | [0.3878, 0.6735] | 49/49 | 0.5306 | 0 / 0 |
| gemma-4-e2b-it | ovd | F1@0.5（ovd_dev/refcoco_dev_images） | 0.2796 | [0.2546, 0.3073] | 336/500 | 0.4170 | 1 / 163 |
| gemma-4-e2b-it | rel_depth_nyu | accuracy（all） | 0.5000 | [0.4360, 0.5600] | 250/250 | 0.5000 | 0 / 0 |
| gemma-4-e2b-it | rel_depth_blink | accuracy（all） | 0.5000 | [0.2500, 0.7500] | 12/12 | 0.5000 | 0 / 0 |
| gemma-4-e2b-it | spatial_cvbench | accuracy（all） | 0.3400 | [0.2000, 0.4600] | 50/50 | 0.3400 | 0 / 0 |
| gemma-4-e2b-it | spatial_embspatial | accuracy（all） | 0.2600 | [0.1400, 0.3805] | 50/50 | 0.2600 | 0 / 0 |
| internvl3.5-2b-hf | refexp_box | acc@0.5（all） | 0.7300 | [0.6933, 0.7650] | 600/600 | 0.7300 | 0 / 0 |
| internvl3.5-2b-hf | refexp_poly | cIoU（all） | 0.0890 | [0.0672, 0.1127] | 388/500 | 0.1096 | 0 / 112 |
| internvl3.5-2b-hf | refexp_poly | gIoU（all） | 0.0588 | [0.0492, 0.0684] | 388/500 | 0.0757 | 0 / 112 |
| internvl3.5-2b-hf | grid16 | gIoU（all） | 0.0000 | [0.0000, 0.0000] | 2/500 | 0.0000 | 0 / 498 |
| internvl3.5-2b-hf | grid16 | cIoU（all） | 0.0000 | [0.0000, 0.0000] | 2/500 | 0.0000 | 0 / 498 |
| internvl3.5-2b-hf | count | accuracy（all） | 0.5714 | [0.4490, 0.6939] | 45/49 | 0.6222 | 0 / 4 |
| internvl3.5-2b-hf | ovd | F1@0.5（ovd_dev/refcoco_dev_images） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| internvl3.5-2b-hf | rel_depth_nyu | accuracy（all） | 0.5080 | [0.4440, 0.5681] | 250/250 | 0.5080 | 0 / 0 |
| internvl3.5-2b-hf | rel_depth_blink | accuracy（all） | 0.5000 | [0.2500, 0.7500] | 12/12 | 0.5000 | 0 / 0 |
| internvl3.5-2b-hf | spatial_cvbench | accuracy（all） | 0.6800 | [0.5400, 0.8005] | 50/50 | 0.6800 | 0 / 0 |
| internvl3.5-2b-hf | spatial_embspatial | accuracy（all） | 0.5400 | [0.4000, 0.6800] | 50/50 | 0.5400 | 0 / 0 |
| qwen3.5-9b | refexp_box | acc@0.5（all） | 0.8867 | [0.8617, 0.9117] | 592/600 | 0.8986 | 0 / 8 |
| qwen3.5-9b | refexp_poly | cIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-9b | refexp_poly | gIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-9b | grid16 | gIoU（all） | 0.0000 | [0.0000, 0.0000] | 433/500 | 0.0000 | 0 / 67 |
| qwen3.5-9b | grid16 | cIoU（all） | 0.0000 | [0.0000, 0.0000] | 433/500 | 0.0000 | 0 / 67 |
| qwen3.5-9b | count | accuracy（all） | 0.9184 | [0.8367, 0.9796] | 49/49 | 0.9184 | 0 / 0 |
| qwen3.5-9b | ovd | F1@0.5（ovd_dev/refcoco_dev_images） | 0.5859 | [0.5554, 0.6189] | 446/500 | 0.6703 | 0 / 54 |
| qwen3.5-9b | rel_depth_nyu | accuracy（all） | 0.8480 | [0.8040, 0.8880] | 250/250 | 0.8480 | 0 / 0 |
| qwen3.5-9b | rel_depth_blink | accuracy（all） | 0.9167 | [0.7500, 1.0000] | 12/12 | 0.9167 | 0 / 0 |
| qwen3.5-9b | spatial_cvbench | accuracy（all） | 0.7200 | [0.6000, 0.8400] | 50/50 | 0.7200 | 0 / 0 |
| qwen3.5-9b | spatial_embspatial | accuracy（all） | 0.7800 | [0.6600, 0.9000] | 50/50 | 0.7800 | 0 / 0 |

#### 拒答诊断：定位、多边形、网格按 person 与非 person 分列（A1 的 refusal_by_person）

| 模型 | 任务 | 指标 | 类 | n | 拒答 | 拒答率 | 有效 | 严格 | 条件 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | acc@0.5 | person | 283 | 0 | 0.000 | 280 | 0.8622 | 0.8714 |
| qwen3.5-0.8b | refexp_box | acc@0.5 | non_person | 317 | 0 | 0.000 | 313 | 0.7603 | 0.7700 |
| qwen3.5-0.8b | refexp_poly | cIoU | person | 238 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-0.8b | refexp_poly | cIoU | non_person | 262 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-0.8b | grid16 | gIoU | person | 243 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-0.8b | grid16 | gIoU | non_person | 257 | 0 | 0.000 | 0 | 0.0000 | — |
| gemma-4-e2b-it | refexp_box | acc@0.5 | person | 283 | 176 | 0.622 | 52 | 0.1625 | 0.8846 |
| gemma-4-e2b-it | refexp_box | acc@0.5 | non_person | 317 | 106 | 0.334 | 99 | 0.1987 | 0.6364 |
| gemma-4-e2b-it | refexp_poly | cIoU | person | 238 | 0 | 0.000 | 237 | 0.2539 | 0.2546 |
| gemma-4-e2b-it | refexp_poly | cIoU | non_person | 262 | 0 | 0.000 | 257 | 0.2427 | 0.2455 |
| gemma-4-e2b-it | grid16 | gIoU | person | 243 | 0 | 0.000 | 0 | 0.0000 | — |
| gemma-4-e2b-it | grid16 | gIoU | non_person | 257 | 0 | 0.000 | 0 | 0.0000 | — |
| internvl3.5-2b-hf | refexp_box | acc@0.5 | person | 283 | 0 | 0.000 | 283 | 0.7774 | 0.7774 |
| internvl3.5-2b-hf | refexp_box | acc@0.5 | non_person | 317 | 0 | 0.000 | 317 | 0.6877 | 0.6877 |
| internvl3.5-2b-hf | refexp_poly | cIoU | person | 238 | 0 | 0.000 | 176 | 0.0875 | 0.1138 |
| internvl3.5-2b-hf | refexp_poly | cIoU | non_person | 262 | 0 | 0.000 | 212 | 0.0905 | 0.1061 |
| internvl3.5-2b-hf | grid16 | gIoU | person | 243 | 0 | 0.000 | 2 | 0.0000 | 0.0000 |
| internvl3.5-2b-hf | grid16 | gIoU | non_person | 257 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-9b | refexp_box | acc@0.5 | person | 283 | 0 | 0.000 | 281 | 0.9258 | 0.9324 |
| qwen3.5-9b | refexp_box | acc@0.5 | non_person | 317 | 0 | 0.000 | 311 | 0.8517 | 0.8682 |
| qwen3.5-9b | refexp_poly | cIoU | person | 238 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-9b | refexp_poly | cIoU | non_person | 262 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-9b | grid16 | gIoU | person | 243 | 0 | 0.000 | 197 | 0.0000 | 0.0000 |
| qwen3.5-9b | grid16 | gIoU | non_person | 257 | 0 | 0.000 | 236 | 0.0000 | 0.0000 |

#### v2 的三种一致口径（A1 与 B 之间，只作参考；tasks[].reference 原样）

| 模型 | 任务 | 逐位 | IoU≥0.9 | 判定结果 | raw_text 相同 | 指标 | A1 | B | 差 | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | refexp_box | 0.8017 | 0.9983 | 0.9983 | 0.7983 | acc@0.5 | 0.8083 | 0.8100 | -0.0017 | [-0.0050, 0.0000] |
| qwen3.5-0.8b | refexp_poly | 1.0000 | 1.0000 | 1.0000 | 0.8700 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-0.8b | grid16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-0.8b | count | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.7959 | 0.7959 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-0.8b | ovd | 0.9480 | 0.9740 | 0.9740 | 0.9000 | F1@0.5 | 0.2573 | 0.2539 | 0.0034 | [-0.0031, 0.0108] |
| qwen3.5-0.8b | rel_depth_nyu | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.6760 | 0.6760 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-0.8b | rel_depth_blink | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5833 | 0.5833 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-0.8b | spatial_cvbench | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.4800 | 0.4800 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-0.8b | spatial_embspatial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5600 | 0.5600 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e2b-it | refexp_box | 0.8850 | 0.9967 | 0.9983 | 0.6817 | acc@0.5 | 0.1817 | 0.1800 | 0.0017 | [0.0000, 0.0050] |
| gemma-4-e2b-it | refexp_poly | 0.6120 | 0.6120 | 0.6120 | 0.4780 | cIoU | 0.2478 | 0.2443 | 0.0035 | [-0.0057, 0.0127] |
| gemma-4-e2b-it | grid16 | 1.0000 | 1.0000 | 1.0000 | 0.9560 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e2b-it | count | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5306 | 0.5306 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e2b-it | ovd | 0.5080 | 0.8400 | 0.8920 | 0.4560 | F1@0.5 | 0.2796 | 0.2823 | -0.0027 | [-0.0099, 0.0034] |
| gemma-4-e2b-it | rel_depth_nyu | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e2b-it | rel_depth_blink | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e2b-it | spatial_cvbench | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.3400 | 0.3400 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e2b-it | spatial_embspatial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.2600 | 0.2600 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-2b-hf | refexp_box | 0.7800 | 0.9850 | 0.9950 | 0.7800 | acc@0.5 | 0.7300 | 0.7283 | 0.0017 | [-0.0033, 0.0083] |
| internvl3.5-2b-hf | refexp_poly | 0.7660 | 0.7660 | 0.7660 | 0.6860 | cIoU | 0.0890 | 0.0947 | -0.0056 | [-0.0141, 0.0015] |
| internvl3.5-2b-hf | grid16 | 0.9960 | 0.9960 | 0.9960 | 0.9560 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-2b-hf | count | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5714 | 0.5714 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-2b-hf | ovd | 1.0000 | 1.0000 | 1.0000 | 0.9780 | F1@0.5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-2b-hf | rel_depth_nyu | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5080 | 0.5080 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-2b-hf | rel_depth_blink | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-2b-hf | spatial_cvbench | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.6800 | 0.6800 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-2b-hf | spatial_embspatial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5400 | 0.5400 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-9b | refexp_box | 0.4300 | 0.9867 | 1.0000 | 0.4167 | acc@0.5 | 0.8867 | 0.8867 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-9b | refexp_poly | 1.0000 | 1.0000 | 1.0000 | 0.3220 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-9b | grid16 | 0.9720 | 0.9720 | 0.9720 | 0.9640 | gIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-9b | count | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.9184 | 0.9184 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-9b | ovd | 0.1040 | 0.5940 | 0.7700 | 0.0100 | F1@0.5 | 0.5859 | 0.5742 | 0.0117 | [-0.0026, 0.0266] |
| qwen3.5-9b | rel_depth_nyu | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.8480 | 0.8480 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-9b | rel_depth_blink | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.9167 | 0.9167 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-9b | spatial_cvbench | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.7200 | 0.7200 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-9b | spatial_embspatial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.7800 | 0.7800 | 0.0000 | [0.0000, 0.0000] |

#### 计算量（compute.json，9 个任务合计；墙钟只含 generate）与显存峰值（torch：验收不记录；nvidia-smi：按进程，每 5 s）

| 模型 | A1 生成 token / 墙钟 s | A2 生成 token / 墙钟 s | B 生成 token / 墙钟 s | 三遍合计墙钟 s | torch 峰值 | nvidia-smi 峰值 MiB |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | 1178887 / 480.0 | 1178887 / 467.4 | 1184912 / 615.7 | 1563.0 | 未记录 | 9596 |
| gemma-4-e2b-it | 274072 / 282.6 | 274072 / 281.9 | 272689 / 347.3 | 911.7 | 未记录 | 16588 |
| internvl3.5-2b-hf | 385390 / 568.2 | 385390 / 442.8 | 384508 / 580.1 | 1591.1 | 未记录 | 26964 |
| qwen3.5-9b | 456192 / 730.1 | 456192 / 715.7 | 465312 / 820.6 | 2266.4 | 未记录 | 34550 |

## 4. 与任务书假设不符的发现、需要管理员定的事

- **qwen3.5-9b 不达标后没有扩到全部 8 个模型（请管理员决定）。**
  - 任务书"为什么做"写的是"按 D38-5 分两步，先跑 4 个模型……不达标再扩到全部 8 个"；但 §3 只给了这 4 个模型的顺序与命令，没有像 RUN-0012 §3 那样写明补跑其余 4 个。
  - 补跑会再用约 2 GPU 小时的 E1 预算（按本次每个模型 17–40 min 估）。
  - 所以我只跑了任务书列出的 4 个。若要扩，其余 4 个是 qwen3.5-2b、qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf，用同一条命令即可，`--out` 相同，各模型有自己的子目录。
- **qwen3.5-9b 检测组的细目（只陈述，见 §3 的 v2 参考列）：**
  - A1 与 B 的 raw_text 逐字相同的只有 0.010（5/500）；
  - 逐位一致 0.104，IoU≥0.9 一致 0.594，判定结果一致 0.770；
  - A1 有效 446/500。
- **torch 峰值显存：** run_e1 的验收不记录，表里只有 nvidia-smi 的按进程峰值。
- **用时：** 任务书预计 3–5 GPU 小时，实际 1.89 h。

## 5. 生成的文件

`experiments/E1/0020_acceptance_v3/`（59 MB，888 个文件）：

- 4 个模型的 `acceptance/`，含 A1、A2、B 三遍的全部输出；
- `accept_v3_summary.md/json`、`README.md`、`disk_ledger.tsv`、`pip_dryrun.txt`；
- `logs/`、`probe_scripts/`。

一次推回。

## 6. 磁盘与内存盘收支（`disk_ledger.tsv`，单位 GB，十进制）

| 时间（UTC） | 节点 | 根盘可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 08:59:42 | 0020 start | 38.182 | 3.206 | 4.556 |
| 08:59:42 | before download qwen3.5-0.8b | 38.182 | 3.206 | 4.556 |
| 08:59:43 | after download qwen3.5-0.8b | 38.182 | 3.206 | 4.556 |
| 09:27:22 | after accept qwen3.5-0.8b | 38.153 | 3.207 | 4.582 |
| 09:27:22 | before download gemma-4-e2b-it | 38.153 | 3.207 | 4.582 |
| 09:27:46 | after download gemma-4-e2b-it | 38.153 | 13.487 | 4.582 |
| 09:44:47 | after accept gemma-4-e2b-it | 38.139 | 13.487 | 4.594 |
| 09:44:47 | after delete gemma-4-e2b-it | 38.139 | 3.208 | 4.594 |
| 09:44:48 | before download internvl3.5-2b-hf | 38.139 | 3.208 | 4.594 |
| 09:45:01 | after download internvl3.5-2b-hf | 38.139 | 7.923 | 4.594 |
| 10:12:26 | after accept internvl3.5-2b-hf | 38.118 | 7.923 | 4.609 |
| 10:12:27 | after delete internvl3.5-2b-hf | 38.118 | 3.208 | 4.609 |
| 10:12:27 | before download qwen3.5-9b | 38.118 | 3.208 | 4.609 |
| 10:13:09 | after download qwen3.5-9b | 38.118 | 22.525 | 4.609 |
| 10:53:04 | after accept qwen3.5-9b | 38.087 | 22.525 | 4.635 |
| 10:53:05 | after delete qwen3.5-9b | 38.087 | 3.209 | 4.635 |
| 10:54:14 | 0020 end (worktree removed) | 38.081 | 3.077 | 4.634 |

- **根盘：** 可用空间从 38.182 GB 降到 38.081 GB，全程远高于 3 GB 红线。
- **`$PROJ`：** 从 4.556 GB 升到 4.634 GB（+78 MB，主要是本运行目录的 59 MB），在 15 GB 份额以内（D28-8）。
- **内存盘：** 峰值 22.525 GB，低于 25 GB 上限。结束时 `/dev/shm/visint` 是 3.077 GB，hf-hub 只剩 Qwen3.5-0.8B；worktree 已删除。
