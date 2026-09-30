[RUN-TASK 0008] E5 生成器第二部分（a、b，3D）的跨平台逐位复现核对

## 为什么做

- DEV-0010 合入了 E5 的 a（出平面三维旋转）与 b（视角变化）。
- 渲染是自写的整数光栅化。按 D24/D27，a、b 必须在任何平台上逐位相同。
- 钉住的摘要只在开发机（arm64 macOS）上生成过。这次在服务器上核对。

## 0. 先读

- `CLAUDE.md`、`prompts/server/STANDBY.md`、`.claude/skills/exp-triad-flow/references/server.md`
- `DECISIONS.md` 的 D24、D25、D27
- `scripts/synth_pin_digests.py --help`

本任务只运行，不改任何仓库代码，也不改 `tests/data/`。

## 1. 拉取与校验

`git pull --ff-only`，HEAD 必须包含 `bd23fc0` 与本任务书所在的 commit。不包含就停下报告。

## 2. 环境

不装任何包：`requirements.txt` 自 RUN-0007 以来没有变。先确认这一点：跑 `pip install --dry-run -r requirements.txt`，应当没有任何 "Would install"。

## 3. 核对

1. **全量测试**：`python -m pytest tests/ -q -rsx`，完整输出存进 `pytest.txt`。
   - 预期：除 11 个 skip（10 个缺 punkt_tab 数据、1 个缺 cv2）外，全部通过。
   - 任何失败都**停下报告**，贴出完整 traceback，不改代码。
2. **摘要核对**：`python scripts/synth_pin_digests.py check`，原样存进 `synth_check.txt`。
   - 预期五个任务都是 60/60 相同；
   - d 在本平台（规范平台）上计入。
3. **有不同时**：
   - 不改文件；
   - 用 `write --out experiments/P0/0008_synth3d_check/synth_digests_server.json` 写出本平台的全部摘要；
   - 对不同的 case，用 `src.synth` 的公开接口算出各分量的子摘要（图像、目标、ID 图、深度、各基线、元数据），写进 `component_digests.json`，供管理员在开发机上逐项对照。

## 4. 产出与回传

- `experiments/P0/0008_synth3d_check/`：`pip_dryrun.txt`、`pytest.txt`、`synth_check.txt`、`README.md`；有不同时再加上面两个文件。
- `reports/0008-synth3d-check.md`：
  - 按 report 模板，只贴数字与事实，不做解读；
  - 第 6 节写磁盘收支，预期没有变化。

一批推完：

```bash
git add <逐个路径> && git commit -m "[results] 0008: <摘要>" && git push
```

完成后按 STANDBY.md，把 `0008-synth3d-check.md` 追加进完成清单，重新进入值守。
