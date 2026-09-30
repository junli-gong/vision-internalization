[RUN-TASK 0006] 服务器环境更新（pymunk）+ 全量测试 + E5 生成器的跨机器复现核对与生成速度

## 为什么做

- DEV-0006（解析层）与 DEV-0007（E5 合成数据生成器第一部分）已合入 main。
- DEV-0007 新增依赖 pymunk（2D 物理引擎），服务器 venv 还没装：不装的话，服务器上的 pytest 会红。
- E5 生成器要求同一样本在任何机器上逐字节相同。c（遮挡补全）与 e（文本到意象）预期能做到；d（短时物理）依赖 Chipmunk 的编译浮点行为，开发机是 arm64、服务器是 x86_64，可能不同。
- `tests/test_synth_repro.py` 里钉住的样本摘要，就是这次的跨机器核对。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D22、D23
- `tests/test_synth_repro.py` 开头的文档字符串：写明了钉住的摘要只在服务器上红时该怎么办

本任务只运行，不改任何仓库代码。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `5b59be7`。不包含就停下报告。

## 2. 环境更新

1. 在 venv 里跑 `pip install --dry-run -r requirements.txt`，结果存进 `pip_dryrun.txt`。
   - 预期只新装 pymunk 及其依赖（cffi、pycparser），不升降任何已装的包。
   - 会改动已装包就**停下报告**，不要装。
2. 通过就执行 `PIP_NO_CACHE_DIR=1 pip install -r requirements.txt`。
3. 安装前后各记一次 `df /` 与 `$PROJ` 的占用。

## 3. 全量测试

用 venv 跑 `python -m pytest tests/ -q`，完整输出存进 `pytest.txt`。

- **预期**：852 条测试，服务器没装 cv2，所以是 851 passed + 1 skipped。
- **`tests/test_synth_repro.py::test_pinned_digests_physics`（以及 `..._occlusion_and_imagery`）失败**：这是本任务要核对的结果之一，不是停下的条件。
  - **不要改摘要**；
  - 把失败信息里"实际得到的摘要"与"钉住的摘要"原样写进 report；
  - 继续做第 4 节。
- **其他任何测试失败**：停下报告完整 traceback，不改代码。

## 4. 跨机器核对与生成速度

写一个探针脚本，放在 `experiments/P0/0006_env_synth/probe_scripts/`，只调用 `src.synth` 的公开接口。

1. **摘要**：对 `tests/test_synth_repro.py` 里 `PINNED_CE` 与 `PINNED_D` 的每个 case，用 `generate(*case).digest()` 算出服务器上的摘要，写进 `synth_digests.json`，并逐条标出与钉住的值是否相同。
2. **d 不一致时的定位**：对不一致的 case，额外报告：
   - 是哪一步开始出现差异：同一 case 在 k 不同的样本里比较物体位姿；
   - 位姿最大差（像素与角度）；
   - 目标 mask 的 IoU。

   这些数用来判断差异是否只在浮点末位。
3. **速度**：c、d、e 各生成 200 个样本（档位按默认轮转），单进程。报告每类的样本/秒，CPU 时间与墙钟都给，写进 `speed.json`。

## 5. 产出

- `experiments/P0/0006_env_synth/`：
  - `pip_dryrun.txt`、`pytest.txt`
  - `synth_digests.json`、`speed.json`
  - `disk_ledger.tsv`、`README.md`、`probe_scripts/`
- `reports/0006-env-synth.md`，按 report 模板：
  - 第 3 节原样贴测试结果、摘要比对表与速度表；
  - 第 6 节写磁盘收支；
  - 只贴数字与事实，不做解读。

## 6. 回传

一批推完：

```bash
git add <逐个路径> && git commit -m "[results] 0006: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0006-env-synth.md` 追加进完成清单，重新进入值守。
