# experiments/P0/0001_env_probe — RUN-0001 证据

b200 服务器 CC，2026-09-30。代码 commit：本任务书 `587b9a5`，外加本批 results commit。本目录不含实验逻辑，只有 Phase 0 探针。
结论性文字不在这里，事实汇总在 `docs/env_report.md`、`docs/model_notes.md`、`docs/data_sources.md`，运行报告在 `reports/0001-phase0-env.md`。

## probe_scripts/（在服务器上从 `$PROJ/scripts/` 运行，这里是副本）

| 文件 | 作用 | 对应节 |
|---|---|---|
| `env.sh` | 所有探针 source 的环境（HF_HOME、HF_HUB_CACHE=/dev/shm、TMPDIR、xet 缓存、pip 不留缓存） | §3 |
| `diskcheck.sh` | 记录 `df /`、`/dev/shm/visint`、`$PROJ` 体积到 `raw/disk_ledger.tsv`；根盘 < 3 GB 时 exit 3 | §2 |
| `p3_dryrun.sh`、`p3_dryrun_sgl_norust.sh` | vLLM nightly / SGLang main 的 pip dry-run（临时目录在 /dev/shm） | §3.4 |
| `p3_dryrun_sizes.py` | 对 pip `--report` 里的每个 URL 发 HTTP HEAD，求下载体积；列出 torch 相关依赖 | §3.4 |
| `p4_model_registry.py`、`p4_table.py` | HF API 登记候选仓库（sha、license、gated、体积、config 摘要），渲染成表 | §4 |
| `p5_facts.py` | Qwen3.5 结构、词表/嵌入行、新增 token、chat template、图像 token 化、M-RoPE 位置 | §5.1–5.4 |
| `p5_5_box_zeroshot.py` | 2 张 COCO val2017 图各做一次零样本框 | §5.5 |
| `remote_zip.py` | 用 HTTP Range 从远端 zip 中只取一个成员（COCO 标注） | §5.5、§6 |
| `p5_6_gdn_kernels.py` | GDN 六种实现加 fp64 参考的两两误差 | §5.6.1 |
| `p5_6_model_paths.py` | 0.8B 整模型 fla 路径与 torch 回退的 logits 对比（`inproc`；`blockfla` 为屏蔽 fla 的独立进程） | §5.6.2–3 |
| `p5_7_taskmod.py` | 图在前 / 指令在前时，图像 token 位置各层隐藏状态（D1 仪器） | §5.7 |
| `p5_8_mem.py` | 显存峰值与 generate 吞吐（`infer`、`lora`） | §5.8 |
| `p5_dl_9b.sh` | 下载 Qwen3.5-9B 到内存盘 | §5 |
| `p5_render.py` | 从 JSON 渲染 docs 里的表格（报告中的数字都来自它） | — |
| `p6_*.py` | 数据源体积与 schema 探针（第 3 批加入） | §6 |

## raw/

| 路径 | 内容 |
|---|---|
| `disk_ledger.tsv` | 每一步的根盘可用字节、`/dev/shm/visint` 字节、`$PROJ` 字节 |
| `env_versions.txt` | Python、torch、CUDA、各包版本 |
| `venv_create.log`、`pip_install_batch{1,2}.log`、`pip_dryrun_batch{1,2}.json` | venv 创建与两批安装 |
| `pip_dryrun_vllm_nightly.{log,json}`、`*_sizes.txt` | vLLM dry-run 原始输出、pip report、HEAD 体积 |
| `pip_dryrun_sglang_main.log` | SGLang 第一次 dry-run（因缺 cargo 失败） |
| `pip_dryrun_sglang_main_norust.{log,json}`、`*_sizes.txt` | SGLang 加 `SGLANG_BUILD_RUST_EXTS=none` 重跑 |
| `p4/registry.json`、`p4/registry_table.md` | 32 个候选仓库的登记数据 |
| `p4/<org>__<repo>/` | 各仓库的 config.json、generation_config.json、preprocessor/processor config、tokenizer_config.json、chat_template.jinja（按登记时的 sha 下载；README 和 LICENSE 不入库，按 sha 链接可复现） |
| `p5/qwen3.5-{0.8b,9b}/facts.json` | §5.1–5.4 的全部原始事实（含各种调用方式渲染出的完整模板文本） |
| `p5/qwen3.5-{0.8b,9b}/model_repr.txt` | `repr(model)` |
| `p5/qwen3.5-{0.8b,9b}/box_zeroshot.json` | §5.5 原始输出（含特殊 token） |
| `p5/coco_gt_2images.json` | 两张图的 COCO val2017 GT（xywh 像素） |
| `p5/gdn_kernels.json` | §5.6.1 两两误差 |
| `p5/gdn_model_paths_{inproc,blockfla}.json` | §5.6.2–3 |
| `p5/qwen3.5-*/taskmod_{bf16,fp32}.json` | §5.7 每层 equal / max\|Δ\| / cos |
| `p5/qwen3.5-*/mem_{infer,lora}.json` | §5.8 |
| `p5/render.json` | 渲染进 docs 的片段 |
| `p5_8_nvsmi.csv`、`p5_8_marks.txt` | §5.8 期间 nvidia-smi 每 500 ms 一次的采样，以及各窗口的起止时间 |
| `*.log` | 各探针的 stdout/stderr（原样） |
| `p6/` | 数据源探针原始输出（第 3 批加入） |
