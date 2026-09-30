[RUN-TASK 0003] G0 冒烟（Qwen3.5）：服务器全量测试 + 0.8B 与 4B 的适配器冒烟

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D6、D9、D13、D15
- `docs/model_notes.md` §5
- `scripts/smoke_adapter.py` 开头的说明

本任务只运行，不改任何仓库代码。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `a4a60e8`（DEV-0003 合入的 merge commit）。不包含就停下报告。

## 2. 环境

- 在 `$PROJ/venv` 里跑 `pip install --dry-run -r requirements.txt`，确认不会改动任何已装包（transformers==5.17.0 应已装好）。会改动就停下报告，不要装。
- 其余照旧：环境块每行一个 export；D6 磁盘红线；内存盘不超过 25 GB。

## 3. 服务器上跑全量测试（检查跨机器差异）

- 用 venv 跑 `pytest tests/ -q`。开发机上的结果是 **312 passed**，原样贴出服务器的结果和耗时。
- 测试会从 HF 下载固定 revision 的 Qwen3.5-0.8B 小文件（tokenizer、config、processor），不含权重。
- **任何失败都属于"开发机绿、算力机红"**：停下，推回完整 traceback 和与开发机的环境差异（torch 2.13 与 2.14、CUDA 可用与否、torchvision 有无），不要改代码。

## 4. 适配器冒烟（真实权重）

测试图用 COCO val2017 的 `000000039769.jpg`，与 RUN-0001 是同一张。按 URL 单张下载到 `$PROJ/tmp`。

1. **0.8B**：

   ```bash
   python scripts/smoke_adapter.py --model qwen3.5-0.8b --device cuda --dtype bfloat16 \
     --image $PROJ/tmp/000000039769.jpg --out experiments/P0/0003_g0_smoke_qwen/smoke_qwen3.5-0.8b.json
   ```

2. **4B**：用来核对 `tie_word_embeddings`，这是 D15 更正 D9-3 的依据。
   - 先按 D13-6 的方法，用 `scan_cache_dir(...).delete_revisions(...)` 从内存盘删掉 gemma-4-E2B-it，腾出空间（RUN-0004 需要时再下）。
   - 按 registry 里的 revision 下载 Qwen3.5-4B，同样跑一遍，`--out experiments/P0/0003_g0_smoke_qwen/smoke_qwen3.5-4b.json`。
   - 跑完删掉 4B，E1 之前用不到。
   - 内存盘占用全程不超过 25 GB，每一步前后记进磁盘收支。
3. 冒烟脚本以非零码退出时：JSON 已经落盘，原样推回，并在 report 里贴出失败项与完整报错，不要改代码。

## 5. 产出

- `experiments/P0/0003_g0_smoke_qwen/`：两份冒烟 JSON、`pytest.txt`（全量测试的原始输出）、`README.md`。
- `reports/0003-g0-smoke-qwen.md`，按 report 模板。第 3 节原样贴两份 JSON 里各检查项的结果；第 6 节必须有磁盘与内存盘收支。

## 6. 回传

分两批：先推 §3 的全量测试结果，再推 §4 的冒烟结果和 report。

```bash
git add <逐个路径> && git commit -m "[results] 0003: <本批摘要>" && git push
```

完成后按 STANDBY.md，把 `0003-g0-smoke-qwen.md` 追加进完成清单，重新进入值守。
