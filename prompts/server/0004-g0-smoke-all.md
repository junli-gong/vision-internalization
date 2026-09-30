[RUN-TASK 0004] G0 冒烟（三家族）：服务器全量测试 + Qwen3.5-0.8B / Gemma 4 E2B-it / InternVL3.5-2B-HF 真实权重冒烟 + Qwen3.5-4B

取代 RUN-0003 §3–4。RUN-0003 暴露的跨机器差异已由 DEV-0003-a1 修复，见 D18。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D6、D9、D13、D15、D18、D19
- `scripts/smoke_adapter.py` 开头的说明：参数以脚本的 argparse 为准。`--dtype` 只接受 `bf16` 或 `fp32`；`--model` 用注册表的键

本任务只运行，不改任何仓库代码。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `b4b4810`。不包含就停下报告。

## 2. 环境

- venv 里跑 `pip install --dry-run -r requirements.txt`，确认不会改动任何已装包。会改动就停下报告。
- 其余照旧：环境块每行一个 export；D6 磁盘红线；内存盘不超过 25 GB。

## 3. 服务器全量测试

- 用 venv 跑 `python -m pytest tests/ -q`。开发机普通口径与服务器口径（parity 脚本）都是 **559 passed**。服务器没装 cv2，预期 **558 passed + 1 skipped**。
- 不需要在服务器上跑 `scripts/pytest_server_parity.sh`：它是给开发机模拟服务器用的。
- **有任何失败都停下**：推回完整 traceback 与环境差异，不改代码，不往下跑冒烟。

## 4. 真实权重冒烟

- 测试图：COCO val2017 `000000039769.jpg`，按 URL 单张下载到 `$PROJ/tmp`。RUN-0003 已下过就直接用。
- 输出目录：`experiments/P0/0004_g0_smoke/`。
- 四条命令都在 `$REPO` 下执行，依次跑：

```bash
python scripts/smoke_adapter.py --model qwen3.5-0.8b --device cuda --dtype bf16 \
  --image $PROJ/tmp/000000039769.jpg --out experiments/P0/0004_g0_smoke/smoke_qwen3.5-0.8b.json
python scripts/smoke_adapter.py --model gemma-4-e2b-it --device cuda --dtype bf16 \
  --image $PROJ/tmp/000000039769.jpg --out experiments/P0/0004_g0_smoke/smoke_gemma-4-e2b-it.json
python scripts/smoke_adapter.py --model internvl3.5-2b-hf --device cuda --dtype bf16 \
  --image $PROJ/tmp/000000039769.jpg --think-max-new-tokens 2048 \
  --out experiments/P0/0004_g0_smoke/smoke_internvl3.5-2b-hf.json
```

- 然后跑 **Qwen3.5-4B**，用来核对 `tie_word_embeddings`（D15）。
  1. 先按 D13-6 的方法，用 `scan_cache_dir(...).delete_revisions(...)` 从内存盘删掉 gemma-4-E2B-it，腾出空间。
  2. 按 registry 的 revision 下载 4B，照同样的方式跑，`--out experiments/P0/0004_g0_smoke/smoke_qwen3.5-4b.json`。
  3. 跑完删掉 4B。
  4. 内存盘占用全程不超过 25 GB，每一步前后都记账。
- 冒烟以非零码退出时，JSON 已经落盘：照常推回，在 report 里贴出失败项与报错，继续跑下一个。冒烟失败不算"停下"的条件，全量测试失败才是。

## 5. 产出

- `experiments/P0/0004_g0_smoke/`：
  - 四份冒烟 JSON；
  - `pytest.txt`：全量测试的原始输出；
  - `README.md`。
- `reports/0004-g0-smoke-all.md`，按 report 模板：
  - 第 3 节原样贴每份 JSON 里各检查项的 ok，以及以下关键数值：
    - 检查 0：实际走的内核实现；Qwen 在 GPU 上应为 fla；lm_head 是否共享。
    - 检查 1：与官方 processor 的逐项比对结果。
    - 检查 2：每层是否逐位相同，以及 max|Δ|。
    - 检查 3：grid token 的 id。
    - 检查 7：思考段属于哪种情形，以及 format_facts。
  - 第 6 节必须有磁盘与内存盘的收支。

## 6. 回传

分两批：先推 §3 的全量测试结果，再推 §4 的冒烟结果和 report。

```bash
git add <逐个路径> && git commit -m "[results] 0004: <本批摘要>" && git push
```

完成后按 STANDBY.md，把 `0004-g0-smoke-all.md` 追加进完成清单，重新进入值守。
