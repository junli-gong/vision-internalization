# RUN-0024 — E1 批处理验收 v3 扩到其余 4 个模型（qwen3.5-2b、qwen3.5-4b、gemma-4-e4b-it、internvl3.5-4b-hf）

服务器 CC，b200，2026-10-02 11:11–13:23 UTC，值守中接单。只贴数字与事实，不做解读。

## 0. 一句话观察（只陈述，不解读）

- **总判定：**
  - `all_pass=True` 的是 qwen3.5-4b、internvl3.5-4b-hf。
  - qwen3.5-2b：`rerun_all_identical=False`（refexp_box），`groups_failed=['detection']`。
  - gemma-4-e4b-it：`groups_failed=['polygon']`。
- **(i) 逐位重跑：** 36 个"模型 × 任务"里有 1 个文件不同，是 qwen3.5-2b 的 refexp_box。600 条里只有 1 条不同（`refcoco+_unc/val/7909`）：
  - 第 211 条，第 4 批（批大小 64，从 1 数）；
  - A1 的框是 `[506, 496, 631, 988]`，A2 是 `[508, 496, 631, 988]`；
  - 两遍都判为对；生成的 token 数都是 44；
  - 同一批的其余 63 条两遍逐字节相同。
  - 这段时间 GPU 上只有本任务的一个进程（pid 969824）。
- **(ii) 噪声底：** 20 个"模型 × 组"里 18 个的 95% CI 在 ±0.02 以内。不在的两个：
  - qwen3.5-2b detection：F1@0.5 A1 0.4599、B 0.4632，差 −0.0033，CI [−0.0207, +0.0130]，下界超过 −0.02；
  - gemma-4-e4b-it polygon：cIoU A1 0.0691、B 0.0598，差 +0.0093，CI [−0.0024, +0.0246]，上界超过 +0.02。
- **8 个模型合计（RUN-0020 与本任务，见 §3 的总表）：**
  - all_pass 5/8；
  - (i) 72 个任务文件里 71 个相同；
  - (ii) 40 个组里 37 个通过。不通过的是：
    - qwen3.5-9b detection（RUN-0020）；
    - qwen3.5-2b detection；
    - gemma-4-e4b-it polygon。
- **用时与磁盘：** 累计 7,876 s（2.19 h），没有触到 7 h 上限。根盘 38.099 → 38.028 GB，`$PROJ` 4.632 → 4.690 GB，内存盘峰值 19.295 GB。

## 1. 前置校验

- **HEAD：** `git pull --ff-only` 之后 HEAD 是 `e319047`。`2257e08`（D43）与本任务书所在的 `728e1f8` 都在 HEAD 的历史里。
- **配置：** `configs/e1_prompts.json` 的 sha256 是 `5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158`，与任务书相同；各 acceptance.json 的 header 也是这个值。
- **环境：** `pip install --dry-run -r requirements.txt` 没有任何 "Would install"。环境变量按任务书 §2 设置。
- **运行用的代码：** 全部在固定于 `e319047` 的 worktree 里运行，跑完已删除。
  - 驱动由 RUN-0020 的复制而来，只改了路径（`probe_scripts/run_accept_v3_ext.sh`）。
  - `p25_accept_v3_summary.py` 原样复制。
- **改动范围：** 没有修改 `src/`、`scripts/`、`tests/`、`configs/`，也没有安装 Python 包。
- **HOLD：** 运行期间 origin 上没有 `HOLD` 文件。
- **只用了开发子集。** 没有重跑 RUN-0020 的 4 个模型，也没有换别的批组成。

## 2. 执行记录

| # | 模型 | 开始时累计 s | 权重核对 | run_e1 墙钟（`time` real） | rc | all_pass | nvidia-smi 峰值 MiB | 删除释放 |
|---|---|---|---|---|---|---|---|---|
| 1 | qwen3.5-2b | 0 | 13/13 | 24m38.1s | 0 | False | 11622 | 4.558 GB |
| 2 | gemma-4-e4b-it | 1498 | 9/9 | 24m45.2s | 0 | False | 22258 | 16.025 GB |
| 3 | internvl3.5-4b-hf | 3021 | 20/20 | 44m55.8s | 0 | True | 37198 | 9.483 GB |
| 4 | qwen3.5-4b | 5741 | 14/14 | 35m10.6s | 0 | True | 26104 | 9.330 GB |

- 结束于 13:22:34，累计 7,876 s。
- **GPU：** NVIDIA B200，bf16。nvidia-smi 每 5 s 采样一次。采样期间 GPU 上依次出现 4 个进程：pid 969824、974456、978997、986466，与 4 次 run_e1 一一对应，时间上互不重叠。
- **报错：** 4 次 run_e1 都没有 Traceback。

## 3. 结果表（原样）

### 3.0 8 个模型的总判定（RUN-0020 与本任务的 acceptance.json）

| 模型 | 来源 | all_pass | rerun_all_identical | rerun_not_identical | groups_failed | short_text 差 [CI] | letter 差 [CI] | polygon 差 [CI] | grid 差 [CI] | detection 差 [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | RUN-0020 | 是 | 是 | [] | [] | -0.0015 [-0.0046, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0034 [-0.0031, +0.0108] |
| gemma-4-e2b-it | RUN-0020 | 是 | 是 | [] | [] | +0.0015 [+0.0000, +0.0046] | +0.0000 [+0.0000, +0.0000] | +0.0035 [-0.0057, +0.0127] | +0.0000 [+0.0000, +0.0000] | -0.0027 [-0.0099, +0.0034] |
| internvl3.5-2b-hf | RUN-0020 | 是 | 是 | [] | [] | +0.0015 [-0.0031, +0.0062] | +0.0000 [+0.0000, +0.0000] | -0.0056 [-0.0141, +0.0015] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| qwen3.5-9b | RUN-0020 | 否 | 是 | [] | ['detection'] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0117 [-0.0026, +0.0266] ✗ |
| qwen3.5-2b | RUN-0024 | 否 | 否 | ['refexp_box'] | ['detection'] | +0.0000 [-0.0062, +0.0062] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0033 [-0.0207, +0.0130] ✗ |
| qwen3.5-4b | RUN-0024 | 是 | 是 | [] | [] | +0.0015 [-0.0046, +0.0092] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0004 [-0.0012, +0.0000] | +0.0009 [-0.0045, +0.0083] |
| gemma-4-e4b-it | RUN-0024 | 否 | 是 | [] | ['polygon'] | +0.0015 [-0.0031, +0.0062] | +0.0000 [+0.0000, +0.0000] | +0.0093 [-0.0024, +0.0246] ✗ | +0.0001 [-0.0013, +0.0015] | -0.0046 [-0.0104, +0.0009] |
| internvl3.5-4b-hf | RUN-0024 | 是 | 是 | [] | [] | -0.0015 [-0.0046, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0018 [-0.0053, +0.0089] | -0.0000 [-0.0000, +0.0000] | +0.0005 [-0.0008, +0.0021] |

合计：all_pass 5/8；(i) 任务文件逐字节相同 71/72；(ii) 组通过 37/40。δ 都是 0.02；✗ 表示该组 CI 不在 ±δ 以内。

### 3.1 本任务的 4 个模型（`accept_v3_summary.md`）

由 `probe_scripts/p25_accept_v3_summary.py` 生成。

#### 总判定（acceptance.json）

| 模型 | all_pass | rerun_all_identical | rerun_not_identical | groups_failed | groups_not_evaluated | groups_below_n_min | 批大小 A / B | frozen |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | 否 | 否 | ['refexp_box'] | ['detection'] | [] | [] | 64 / 48 | 是 |
| qwen3.5-4b | 是 | 是 | [] | [] | [] | [] | 64 / 48 | 是 |
| gemma-4-e4b-it | 否 | 是 | [] | ['polygon'] | [] | [] | 64 / 48 | 是 |
| internvl3.5-4b-hf | 是 | 是 | [] | [] | [] | [] | 64 / 48 | 是 |

#### (i) 逐位重跑：A1 与 A2 的 predictions.jsonl（tasks[].rerun）

| 模型 | 任务 | n | 文件逐字节相同 | 逐条不同的条数 | raw_text 不同的条数 | A1 的 sha256（前 16 位） |
|---|---|---|---|---|---|---|
| qwen3.5-2b | refexp_box | 600 | 否 | 1 | 1 | c8b16e4f15343849 |
| qwen3.5-2b | refexp_poly | 500 | 是 | 0 | 0 | 7b985ab2294e21b6 |
| qwen3.5-2b | grid16 | 500 | 是 | 0 | 0 | 20c2bafda7b7fd84 |
| qwen3.5-2b | count | 49 | 是 | 0 | 0 | e985ddac66e01412 |
| qwen3.5-2b | ovd | 500 | 是 | 0 | 0 | 102b1de5b55cd383 |
| qwen3.5-2b | rel_depth_nyu | 250 | 是 | 0 | 0 | 16eb2a21b3dd8ad5 |
| qwen3.5-2b | rel_depth_blink | 12 | 是 | 0 | 0 | 89d564831138d740 |
| qwen3.5-2b | spatial_cvbench | 50 | 是 | 0 | 0 | e4d415a75fca5add |
| qwen3.5-2b | spatial_embspatial | 50 | 是 | 0 | 0 | 45b600b731cc044d |
| qwen3.5-4b | refexp_box | 600 | 是 | 0 | 0 | 7111038d63af6a49 |
| qwen3.5-4b | refexp_poly | 500 | 是 | 0 | 0 | 754526e14cb90f33 |
| qwen3.5-4b | grid16 | 500 | 是 | 0 | 0 | 70007a8d9198460a |
| qwen3.5-4b | count | 49 | 是 | 0 | 0 | 9d6905a42cb5b41e |
| qwen3.5-4b | ovd | 500 | 是 | 0 | 0 | 15d7e057282ede6e |
| qwen3.5-4b | rel_depth_nyu | 250 | 是 | 0 | 0 | ab29ca67a8491c8b |
| qwen3.5-4b | rel_depth_blink | 12 | 是 | 0 | 0 | 89d564831138d740 |
| qwen3.5-4b | spatial_cvbench | 50 | 是 | 0 | 0 | 915aaa26de6e834e |
| qwen3.5-4b | spatial_embspatial | 50 | 是 | 0 | 0 | 9249311649f12790 |
| gemma-4-e4b-it | refexp_box | 600 | 是 | 0 | 0 | 9506f160e03b769b |
| gemma-4-e4b-it | refexp_poly | 500 | 是 | 0 | 0 | 053d0b1f4852c799 |
| gemma-4-e4b-it | grid16 | 500 | 是 | 0 | 0 | 91f4e9854e354466 |
| gemma-4-e4b-it | count | 49 | 是 | 0 | 0 | 013c615da3d0a599 |
| gemma-4-e4b-it | ovd | 500 | 是 | 0 | 0 | 6333616bf22f7d91 |
| gemma-4-e4b-it | rel_depth_nyu | 250 | 是 | 0 | 0 | 55f4143b4c950339 |
| gemma-4-e4b-it | rel_depth_blink | 12 | 是 | 0 | 0 | 9cc8138d83045f3d |
| gemma-4-e4b-it | spatial_cvbench | 50 | 是 | 0 | 0 | d2b4fe346d27446d |
| gemma-4-e4b-it | spatial_embspatial | 50 | 是 | 0 | 0 | 74869963c175a741 |
| internvl3.5-4b-hf | refexp_box | 600 | 是 | 0 | 0 | 77f9b754e3fed369 |
| internvl3.5-4b-hf | refexp_poly | 500 | 是 | 0 | 0 | 9ac9535d9af27b70 |
| internvl3.5-4b-hf | grid16 | 500 | 是 | 0 | 0 | ac20fed162729dd1 |
| internvl3.5-4b-hf | count | 49 | 是 | 0 | 0 | 2fd1e9390e532a1b |
| internvl3.5-4b-hf | ovd | 500 | 是 | 0 | 0 | 8f69a32744d60c15 |
| internvl3.5-4b-hf | rel_depth_nyu | 250 | 是 | 0 | 0 | f237000493425243 |
| internvl3.5-4b-hf | rel_depth_blink | 12 | 是 | 0 | 0 | 4bf0ef77d4c43c83 |
| internvl3.5-4b-hf | spatial_cvbench | 50 | 是 | 0 | 0 | 6c8fed46a94bb508 |
| internvl3.5-4b-hf | spatial_embspatial | 50 | 是 | 0 | 0 | fe1db33286b68c37 |

#### (ii) 噪声底：A1 与 B 按组比较（groups[]；严格读数，配对 bootstrap 95% CI）

| 模型 | 组 | 任务 | 指标 | n（n_min） | A1 | B | 差（A1−B） | 95% CI | δ | CI 在 ±δ 内 | 通过 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | short_text | refexp_box, count | accuracy_pooled | 649（300） | 0.8567 | 0.8567 | 0.0000 | [-0.0062, 0.0062] | 0.02 | 是 | 是 |
| qwen3.5-2b | letter | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | accuracy_pooled | 362（300） | 0.6575 | 0.6575 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-2b | polygon | refexp_poly | cIoU | 500（500） | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-2b | grid | grid16 | gIoU | 500（500） | 0.0003 | 0.0003 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-2b | detection | ovd | F1@0.5 | 500（None） | 0.4599 | 0.4632 | -0.0033 | [-0.0207, 0.0130] | 0.02 | 否 | 否 |
| qwen3.5-4b | short_text | refexp_box, count | accuracy_pooled | 649（300） | 0.8598 | 0.8582 | 0.0015 | [-0.0046, 0.0092] | 0.02 | 是 | 是 |
| qwen3.5-4b | letter | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | accuracy_pooled | 362（300） | 0.5249 | 0.5249 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-4b | polygon | refexp_poly | cIoU | 500（500） | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-4b | grid | grid16 | gIoU | 500（500） | 0.0077 | 0.0081 | -0.0004 | [-0.0012, 0.0000] | 0.02 | 是 | 是 |
| qwen3.5-4b | detection | ovd | F1@0.5 | 500（None） | 0.5528 | 0.5519 | 0.0009 | [-0.0045, 0.0083] | 0.02 | 是 | 是 |
| gemma-4-e4b-it | short_text | refexp_box, count | accuracy_pooled | 649（300） | 0.2296 | 0.2280 | 0.0015 | [-0.0031, 0.0062] | 0.02 | 是 | 是 |
| gemma-4-e4b-it | letter | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | accuracy_pooled | 362（300） | 0.5387 | 0.5387 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| gemma-4-e4b-it | polygon | refexp_poly | cIoU | 500（500） | 0.0691 | 0.0598 | 0.0093 | [-0.0024, 0.0246] | 0.02 | 否 | 否 |
| gemma-4-e4b-it | grid | grid16 | gIoU | 500（500） | 0.0159 | 0.0158 | 0.0001 | [-0.0013, 0.0015] | 0.02 | 是 | 是 |
| gemma-4-e4b-it | detection | ovd | F1@0.5 | 500（None） | 0.4941 | 0.4987 | -0.0046 | [-0.0104, 0.0009] | 0.02 | 是 | 是 |
| internvl3.5-4b-hf | short_text | refexp_box, count | accuracy_pooled | 649（300） | 0.8336 | 0.8351 | -0.0015 | [-0.0046, 0.0000] | 0.02 | 是 | 是 |
| internvl3.5-4b-hf | letter | rel_depth_nyu, rel_depth_blink, spatial_cvbench, spatial_embspatial | accuracy_pooled | 362（300） | 0.7486 | 0.7486 | 0.0000 | [0.0000, 0.0000] | 0.02 | 是 | 是 |
| internvl3.5-4b-hf | polygon | refexp_poly | cIoU | 500（500） | 0.1972 | 0.1955 | 0.0018 | [-0.0053, 0.0089] | 0.02 | 是 | 是 |
| internvl3.5-4b-hf | grid | grid16 | gIoU | 500（500） | 0.0025 | 0.0026 | -0.0000 | [-0.0000, 0.0000] | 0.02 | 是 | 是 |
| internvl3.5-4b-hf | detection | ovd | F1@0.5 | 500（None） | 0.1407 | 0.1402 | 0.0005 | [-0.0008, 0.0021] | 0.02 | 是 | 是 |

#### 三列（A1 的 metrics.json 的 three_columns：严格读数 / 有效率 / 条件读数）

| 模型 | 任务 | 指标（范围） | 严格 | 严格 95% CI | 有效 n_valid/n | 条件 | 无效：拒答 / 其他失败 |
|---|---|---|---|---|---|---|---|
| qwen3.5-2b | refexp_box | acc@0.5（all） | 0.8600 | [0.8333, 0.8883] | 595/600 | 0.8672 | 0 / 5 |
| qwen3.5-2b | refexp_poly | cIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-2b | refexp_poly | gIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-2b | grid16 | gIoU（all） | 0.0003 | [0.0000, 0.0007] | 7/500 | 0.0192 | 0 / 493 |
| qwen3.5-2b | grid16 | cIoU（all） | 0.0012 | [0.0000, 0.0033] | 7/500 | 0.0494 | 0 / 493 |
| qwen3.5-2b | count | accuracy（all） | 0.8163 | [0.6939, 0.9184] | 49/49 | 0.8163 | 0 / 0 |
| qwen3.5-2b | ovd | F1@0.5（ovd_dev/refcoco_dev_images） | 0.4599 | [0.4248, 0.4966] | 357/500 | 0.6778 | 0 / 143 |
| qwen3.5-2b | rel_depth_nyu | accuracy（all） | 0.6960 | [0.6360, 0.7520] | 250/250 | 0.6960 | 0 / 0 |
| qwen3.5-2b | rel_depth_blink | accuracy（all） | 0.5000 | [0.2500, 0.7500] | 12/12 | 0.5000 | 0 / 0 |
| qwen3.5-2b | spatial_cvbench | accuracy（all） | 0.4800 | [0.3400, 0.6200] | 50/50 | 0.4800 | 0 / 0 |
| qwen3.5-2b | spatial_embspatial | accuracy（all） | 0.6800 | [0.5400, 0.8000] | 50/50 | 0.6800 | 0 / 0 |
| qwen3.5-4b | refexp_box | acc@0.5（all） | 0.8600 | [0.8317, 0.8883] | 586/600 | 0.8805 | 0 / 14 |
| qwen3.5-4b | refexp_poly | cIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-4b | refexp_poly | gIoU（all） | 0.0000 | [0.0000, 0.0000] | 0/500 | — | 0 / 500 |
| qwen3.5-4b | grid16 | gIoU（all） | 0.0077 | [0.0032, 0.0135] | 499/500 | 0.0077 | 0 / 1 |
| qwen3.5-4b | grid16 | cIoU（all） | 0.0561 | [0.0261, 0.0876] | 499/500 | 0.0561 | 0 / 1 |
| qwen3.5-4b | count | accuracy（all） | 0.8571 | [0.7551, 0.9592] | 49/49 | 0.8571 | 0 / 0 |
| qwen3.5-4b | ovd | F1@0.5（ovd_dev/refcoco_dev_images） | 0.5528 | [0.5202, 0.5848] | 435/500 | 0.6487 | 0 / 65 |
| qwen3.5-4b | rel_depth_nyu | accuracy（all） | 0.5000 | [0.4360, 0.5600] | 250/250 | 0.5000 | 0 / 0 |
| qwen3.5-4b | rel_depth_blink | accuracy（all） | 0.5000 | [0.2500, 0.7500] | 12/12 | 0.5000 | 0 / 0 |
| qwen3.5-4b | spatial_cvbench | accuracy（all） | 0.4200 | [0.2800, 0.5400] | 50/50 | 0.4200 | 0 / 0 |
| qwen3.5-4b | spatial_embspatial | accuracy（all） | 0.7600 | [0.6400, 0.8605] | 50/50 | 0.7600 | 0 / 0 |
| gemma-4-e4b-it | refexp_box | acc@0.5（all） | 0.2017 | [0.1700, 0.2350] | 146/600 | 0.8288 | 273 / 181 |
| gemma-4-e4b-it | refexp_poly | cIoU（all） | 0.0691 | [0.0494, 0.0923] | 87/500 | 0.3244 | 0 / 413 |
| gemma-4-e4b-it | refexp_poly | gIoU（all） | 0.0514 | [0.0381, 0.0656] | 87/500 | 0.2956 | 0 / 413 |
| gemma-4-e4b-it | grid16 | gIoU（all） | 0.0159 | [0.0102, 0.0227] | 500/500 | 0.0159 | 0 / 0 |
| gemma-4-e4b-it | grid16 | cIoU（all） | 0.0610 | [0.0381, 0.0864] | 500/500 | 0.0610 | 0 / 0 |
| gemma-4-e4b-it | count | accuracy（all） | 0.5714 | [0.4286, 0.6939] | 49/49 | 0.5714 | 0 / 0 |
| gemma-4-e4b-it | ovd | F1@0.5（ovd_dev/refcoco_dev_images） | 0.4941 | [0.4712, 0.5185] | 492/500 | 0.4986 | 0 / 8 |
| gemma-4-e4b-it | rel_depth_nyu | accuracy（all） | 0.5520 | [0.4880, 0.6121] | 250/250 | 0.5520 | 0 / 0 |
| gemma-4-e4b-it | rel_depth_blink | accuracy（all） | 0.4167 | [0.1667, 0.6667] | 12/12 | 0.4167 | 0 / 0 |
| gemma-4-e4b-it | spatial_cvbench | accuracy（all） | 0.5200 | [0.3800, 0.6600] | 50/50 | 0.5200 | 0 / 0 |
| gemma-4-e4b-it | spatial_embspatial | accuracy（all） | 0.5200 | [0.3800, 0.6600] | 50/50 | 0.5200 | 0 / 0 |
| internvl3.5-4b-hf | refexp_box | acc@0.5（all） | 0.8400 | [0.8100, 0.8700] | 600/600 | 0.8400 | 0 / 0 |
| internvl3.5-4b-hf | refexp_poly | cIoU（all） | 0.1972 | [0.1708, 0.2209] | 486/500 | 0.2014 | 0 / 14 |
| internvl3.5-4b-hf | refexp_poly | gIoU（all） | 0.1716 | [0.1547, 0.1863] | 486/500 | 0.1766 | 0 / 14 |
| internvl3.5-4b-hf | grid16 | gIoU（all） | 0.0025 | [0.0005, 0.0051] | 497/500 | 0.0026 | 0 / 3 |
| internvl3.5-4b-hf | grid16 | cIoU（all） | 0.0160 | [0.0010, 0.0362] | 497/500 | 0.0161 | 0 / 3 |
| internvl3.5-4b-hf | count | accuracy（all） | 0.7551 | [0.6327, 0.8571] | 49/49 | 0.7551 | 0 / 0 |
| internvl3.5-4b-hf | ovd | F1@0.5（ovd_dev/refcoco_dev_images） | 0.1407 | [0.1278, 0.1563] | 457/500 | 0.1505 | 0 / 43 |
| internvl3.5-4b-hf | rel_depth_nyu | accuracy（all） | 0.7760 | [0.7240, 0.8241] | 250/250 | 0.7760 | 0 / 0 |
| internvl3.5-4b-hf | rel_depth_blink | accuracy（all） | 0.7500 | [0.5000, 0.9167] | 12/12 | 0.7500 | 0 / 0 |
| internvl3.5-4b-hf | spatial_cvbench | accuracy（all） | 0.7000 | [0.5800, 0.8200] | 50/50 | 0.7000 | 0 / 0 |
| internvl3.5-4b-hf | spatial_embspatial | accuracy（all） | 0.6600 | [0.5200, 0.7800] | 50/50 | 0.6600 | 0 / 0 |

#### 拒答诊断：定位、多边形、网格按 person 与非 person 分列（A1 的 refusal_by_person）

| 模型 | 任务 | 指标 | 类 | n | 拒答 | 拒答率 | 有效 | 严格 | 条件 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | refexp_box | acc@0.5 | person | 283 | 0 | 0.000 | 281 | 0.8834 | 0.8897 |
| qwen3.5-2b | refexp_box | acc@0.5 | non_person | 317 | 0 | 0.000 | 314 | 0.8391 | 0.8471 |
| qwen3.5-2b | refexp_poly | cIoU | person | 238 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-2b | refexp_poly | cIoU | non_person | 262 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-2b | grid16 | gIoU | person | 243 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-2b | grid16 | gIoU | non_person | 257 | 0 | 0.000 | 7 | 0.0005 | 0.0192 |
| qwen3.5-4b | refexp_box | acc@0.5 | person | 283 | 0 | 0.000 | 280 | 0.8763 | 0.8857 |
| qwen3.5-4b | refexp_box | acc@0.5 | non_person | 317 | 0 | 0.000 | 306 | 0.8454 | 0.8758 |
| qwen3.5-4b | refexp_poly | cIoU | person | 238 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-4b | refexp_poly | cIoU | non_person | 262 | 0 | 0.000 | 0 | 0.0000 | — |
| qwen3.5-4b | grid16 | gIoU | person | 243 | 0 | 0.000 | 243 | 0.0094 | 0.0094 |
| qwen3.5-4b | grid16 | gIoU | non_person | 257 | 0 | 0.000 | 256 | 0.0061 | 0.0061 |
| gemma-4-e4b-it | refexp_box | acc@0.5 | person | 283 | 185 | 0.654 | 54 | 0.1696 | 0.8889 |
| gemma-4-e4b-it | refexp_box | acc@0.5 | non_person | 317 | 88 | 0.278 | 92 | 0.2303 | 0.7935 |
| gemma-4-e4b-it | refexp_poly | cIoU | person | 238 | 0 | 0.000 | 43 | 0.0512 | 0.2510 |
| gemma-4-e4b-it | refexp_poly | cIoU | non_person | 262 | 0 | 0.000 | 44 | 0.0860 | 0.3876 |
| gemma-4-e4b-it | grid16 | gIoU | person | 243 | 0 | 0.000 | 243 | 0.0074 | 0.0074 |
| gemma-4-e4b-it | grid16 | gIoU | non_person | 257 | 0 | 0.000 | 257 | 0.0240 | 0.0240 |
| internvl3.5-4b-hf | refexp_box | acc@0.5 | person | 283 | 0 | 0.000 | 283 | 0.8975 | 0.8975 |
| internvl3.5-4b-hf | refexp_box | acc@0.5 | non_person | 317 | 0 | 0.000 | 317 | 0.7886 | 0.7886 |
| internvl3.5-4b-hf | refexp_poly | cIoU | person | 238 | 0 | 0.000 | 236 | 0.1468 | 0.1489 |
| internvl3.5-4b-hf | refexp_poly | cIoU | non_person | 262 | 0 | 0.000 | 250 | 0.2409 | 0.2476 |
| internvl3.5-4b-hf | grid16 | gIoU | person | 243 | 0 | 0.000 | 242 | 0.0011 | 0.0011 |
| internvl3.5-4b-hf | grid16 | gIoU | non_person | 257 | 0 | 0.000 | 255 | 0.0040 | 0.0040 |

#### v2 的三种一致口径（A1 与 B 之间，只作参考；tasks[].reference 原样）

| 模型 | 任务 | 逐位 | IoU≥0.9 | 判定结果 | raw_text 相同 | 指标 | A1 | B | 差 | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | refexp_box | 0.3467 | 0.9883 | 0.9933 | 0.3367 | acc@0.5 | 0.8600 | 0.8600 | 0.0000 | [-0.0067, 0.0067] |
| qwen3.5-2b | refexp_poly | 1.0000 | 1.0000 | 1.0000 | 0.3180 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-2b | grid16 | 1.0000 | 1.0000 | 1.0000 | 0.9880 | gIoU | 0.0003 | 0.0003 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-2b | count | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.8163 | 0.8163 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-2b | ovd | 0.3200 | 0.7220 | 0.8080 | 0.0920 | F1@0.5 | 0.4599 | 0.4632 | -0.0033 | [-0.0207, 0.0130] |
| qwen3.5-2b | rel_depth_nyu | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.6960 | 0.6960 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-2b | rel_depth_blink | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-2b | spatial_cvbench | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.4800 | 0.4800 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-2b | spatial_embspatial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.6800 | 0.6800 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-4b | refexp_box | 0.8067 | 0.9917 | 0.9917 | 0.8050 | acc@0.5 | 0.8600 | 0.8583 | 0.0017 | [-0.0050, 0.0100] |
| qwen3.5-4b | refexp_poly | 1.0000 | 1.0000 | 1.0000 | 0.8740 | cIoU | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-4b | grid16 | 0.9980 | 0.9980 | 0.9980 | 0.9980 | gIoU | 0.0077 | 0.0081 | -0.0004 | [-0.0012, 0.0000] |
| qwen3.5-4b | count | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.8571 | 0.8571 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-4b | ovd | 0.9060 | 0.9520 | 0.9680 | 0.8980 | F1@0.5 | 0.5528 | 0.5519 | 0.0009 | [-0.0045, 0.0083] |
| qwen3.5-4b | rel_depth_nyu | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-4b | rel_depth_blink | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-4b | spatial_cvbench | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.4200 | 0.4200 | 0.0000 | [0.0000, 0.0000] |
| qwen3.5-4b | spatial_embspatial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.7600 | 0.7600 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e4b-it | refexp_box | 0.8850 | 0.9933 | 0.9950 | 0.6933 | acc@0.5 | 0.2017 | 0.2000 | 0.0017 | [-0.0033, 0.0067] |
| gemma-4-e4b-it | refexp_poly | 0.9340 | 0.9340 | 0.9340 | 0.1860 | cIoU | 0.0691 | 0.0598 | 0.0093 | [-0.0024, 0.0246] |
| gemma-4-e4b-it | grid16 | 0.9860 | 0.9860 | 0.9860 | 0.9860 | gIoU | 0.0159 | 0.0158 | 0.0001 | [-0.0013, 0.0015] |
| gemma-4-e4b-it | count | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5714 | 0.5714 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e4b-it | ovd | 0.2360 | 0.8320 | 0.9000 | 0.2360 | F1@0.5 | 0.4941 | 0.4987 | -0.0046 | [-0.0104, 0.0009] |
| gemma-4-e4b-it | rel_depth_nyu | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5520 | 0.5520 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e4b-it | rel_depth_blink | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.4167 | 0.4167 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e4b-it | spatial_cvbench | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5200 | 0.5200 | 0.0000 | [0.0000, 0.0000] |
| gemma-4-e4b-it | spatial_embspatial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.5200 | 0.5200 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-4b-hf | refexp_box | 0.8833 | 0.9950 | 0.9983 | 0.8833 | acc@0.5 | 0.8400 | 0.8417 | -0.0017 | [-0.0050, 0.0000] |
| internvl3.5-4b-hf | refexp_poly | 0.7360 | 0.7360 | 0.7360 | 0.6680 | cIoU | 0.1972 | 0.1955 | 0.0018 | [-0.0053, 0.0089] |
| internvl3.5-4b-hf | grid16 | 0.9920 | 0.9920 | 0.9920 | 0.9900 | gIoU | 0.0025 | 0.0026 | -0.0000 | [-0.0000, 0.0000] |
| internvl3.5-4b-hf | count | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.7551 | 0.7551 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-4b-hf | ovd | 0.9640 | 0.9740 | 0.9800 | 0.9600 | F1@0.5 | 0.1407 | 0.1402 | 0.0005 | [-0.0008, 0.0021] |
| internvl3.5-4b-hf | rel_depth_nyu | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.7760 | 0.7760 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-4b-hf | rel_depth_blink | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.7500 | 0.7500 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-4b-hf | spatial_cvbench | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.7000 | 0.7000 | 0.0000 | [0.0000, 0.0000] |
| internvl3.5-4b-hf | spatial_embspatial | 1.0000 | 1.0000 | 1.0000 | 1.0000 | accuracy | 0.6600 | 0.6600 | 0.0000 | [0.0000, 0.0000] |

#### 计算量（compute.json，9 个任务合计；墙钟只含 generate）与显存峰值（torch：验收不记录；nvidia-smi：按进程，每 5 s）

| 模型 | A1 生成 token / 墙钟 s | A2 生成 token / 墙钟 s | B 生成 token / 墙钟 s | 三遍合计墙钟 s | torch 峰值 | nvidia-smi 峰值 MiB |
|---|---|---|---|---|---|---|
| qwen3.5-2b | 943782 / 420.6 | 943782 / 417.0 | 943319 / 535.5 | 1373.1 | 未记录 | 11622 |
| qwen3.5-4b | 473713 / 635.8 | 473713 / 632.9 | 473731 / 728.5 | 1997.2 | 未记录 | 26104 |
| gemma-4-e4b-it | 291620 / 429.9 | 291620 / 429.7 | 292000 / 511.7 | 1371.3 | 未记录 | 22258 |
| internvl3.5-4b-hf | 491378 / 911.7 | 491378 / 766.2 | 491042 / 920.6 | 2598.5 | 未记录 | 37198 |

## 4. 与任务书假设不符的发现

- **(i) 第一次出现逐位重跑不同。** RUN-0020 的 36 个任务文件全部相同；本任务 qwen3.5-2b 的 refexp_box 有 1 条不同，其余 35 个相同。
  - 同一条在三遍里第一个坐标分别是：A1 506、A2 508、B 508（0–1000 坐标）。
  - 第二个坐标：A1 496、A2 496、B 495。
  - 判定都是对。
  - 那段时间 GPU 上只有本任务的进程。
- **用时：** 任务书预计 3–5 GPU 小时，实际 2.19 h。
- **torch 峰值显存：** run_e1 的验收不记录，表里只有 nvidia-smi 的按进程峰值。

## 5. 生成的文件

`experiments/E1/0024_acceptance_v3_ext/`（59 MB，888 个文件）：

- 4 个模型的 `acceptance/`，含 A1、A2、B 三遍；
- `accept_v3_summary.md/json`、`README.md`、`disk_ledger.tsv`、`pip_dryrun.txt`；
- `logs/`、`probe_scripts/`。

一次推回。

## 6. 磁盘与内存盘收支（`disk_ledger.tsv`，单位 GB，十进制）

| 时间（UTC） | 节点 | 根盘可用 | `/dev/shm/visint` | `$PROJ` |
|---|---|---|---|---|
| 11:11:18 | 0024 start | 38.099 | 3.268 | 4.632 |
| 11:11:18 | before download qwen3.5-2b | 38.099 | 3.268 | 4.632 |
| 11:11:38 | after download qwen3.5-2b | 38.099 | 7.827 | 4.632 |
| 11:36:16 | after accept qwen3.5-2b | 38.080 | 7.828 | 4.649 |
| 11:36:16 | after delete qwen3.5-2b | 38.080 | 3.270 | 4.649 |
| 11:36:17 | before download gemma-4-e4b-it | 38.080 | 3.270 | 4.649 |
| 11:36:52 | after download gemma-4-e4b-it | 38.080 | 19.295 | 4.649 |
| 12:01:38 | after accept gemma-4-e4b-it | 38.066 | 19.295 | 4.662 |
| 12:01:39 | after delete gemma-4-e4b-it | 38.066 | 3.271 | 4.662 |
| 12:01:39 | before download internvl3.5-4b-hf | 38.066 | 3.271 | 4.662 |
| 12:02:02 | after download internvl3.5-4b-hf | 38.066 | 12.754 | 4.662 |
| 12:46:58 | after accept internvl3.5-4b-hf | 38.043 | 12.754 | 4.677 |
| 12:46:59 | after delete internvl3.5-4b-hf | 38.043 | 3.271 | 4.677 |
| 12:46:59 | before download qwen3.5-4b | 38.043 | 3.271 | 4.677 |
| 12:47:22 | after download qwen3.5-4b | 38.043 | 12.602 | 4.677 |
| 13:22:33 | after accept qwen3.5-4b | 38.027 | 12.602 | 4.691 |
| 13:22:34 | after delete qwen3.5-4b | 38.027 | 3.271 | 4.691 |
| 13:24:40 | 0024 end (worktree removed) | 38.028 | 3.080 | 4.690 |

- **根盘：** 可用空间从 38.099 GB 降到 38.028 GB，全程远高于 3 GB 红线。
- **`$PROJ`：** 从 4.632 GB 升到 4.690 GB（+58 MB，基本是本运行目录的 59 MB），在 15 GB 份额以内。
- **内存盘：** 峰值 19.295 GB，低于 25 GB 上限。结束时 `/dev/shm/visint` 是 3.080 GB，hf-hub 只剩 Qwen3.5-0.8B；worktree 已删除。
