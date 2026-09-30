# experiments/P0/0002_family_probe — RUN-0002 证据

b200 服务器 CC，2026-09-30。本目录不含实验逻辑，只有 Phase 0 探针：对 Gemma 4 E2B-it 与 InternVL3.5-2B-HF 做 RUN-0001 §5 的同类事实核查。
事实汇总写在 `docs/model_notes.md` §6（Gemma）、§7（InternVL）和 `docs/env_report.md` §8；运行报告为 `reports/0002-family-probe.md`。

## probe_scripts/（在服务器上从 `$PROJ/scripts/` 运行，这里是副本）

| 文件 | 作用 | 对应节 |
|---|---|---|
| `env.sh`、`diskcheck2.sh` | 环境（同 RUN-0001）；磁盘账目写入 `raw/disk_ledger.tsv`，根盘低于 3 GB 时 exit 3 | §2 |
| `p7_facts.py` | 结构、词表与嵌入行、占位 token、新增 token、chat template、图像 token 化、位置编码、output_hidden_states | §3.1–3.4 |
| `p7_box.py` | 两张 COCO val2017 图的零样本定位（各家族官方提示写法） | §3.5 |
| `p7_taskmod.py` | 图在前 / 指令在前时，图像 token 位置各层隐藏状态（D1 仪器） | §3.6 |
| `p7_mem.py` | 显存峰值、LoRA 单步、generate 吞吐 | §3.7 |
| `p7_render.py`、`p7_nvsmi_summary.py` | 从 JSON 渲染 docs 表格；汇总 nvidia-smi 窗口 | — |

## raw/

| 路径 | 内容 |
|---|---|
| `disk_ledger.tsv`、`shm_before_delete.txt`、`shm_after_delete.txt` | 磁盘与内存盘收支；删除 Qwen3.5-9B 前后内存盘的占用 |
| `repo_file_sizes.txt` | 下载前按固定 revision 查到的两个仓库的文件清单与大小 |
| `gemma4-e2b-it/` | `facts.json`、`model_repr.txt`、`box_zeroshot.json`、`taskmod_bf16.json`、`mem_{infer,lora}.json`、`unused_tokenization.txt`、`render*.{md,json}` |
| `internvl3.5-2b-hf/` | 同上（第 2 批加入），另有 `tile_grids.txt` |
| `docs_evidence/` | ai.google.dev Gemma image understanding 页面与 InternVL chat data format 页面的原始 HTML 及抽出的文本（用于核对坐标格式原文） |
| `p7_*.log`、`p7_mem_nvsmi_*.csv`、`p7_mem_marks.txt` | 各探针的 stdout/stderr；§3.7 期间 nvidia-smi 每 500 ms 一次的采样与各窗口的起止时间 |
