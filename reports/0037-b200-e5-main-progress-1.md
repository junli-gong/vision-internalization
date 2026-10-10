# RUN-0037 进度报告 1 — E5 起点检查点的核对（RUN-0040 跑完之后）

服务器 CC 的核对脚本（与 CC 脱钩），2026-10-10T20:44:22Z。只记事实。任务书 §1：启动时 RUN-0040 还没跑完，起点检查点在它跑完之后核对。

- 9 个起点：`.sha256.txt` 逐文件核对、`model/` 在、叶目录有 `config.yaml`。全部通过：是。

```
e3b_c3 s0 model=y sha256=ok files=5 config.yaml=y kb=131768
e3b_c3 s1 model=y sha256=ok files=5 config.yaml=y kb=131764
e3b_c3 s2 model=y sha256=ok files=5 config.yaml=y kb=131768
c3_warmup s0 model=y sha256=ok files=5 config.yaml=y kb=263172
c3_warmup s1 model=y sha256=ok files=5 config.yaml=y kb=263172
c3_warmup s2 model=y sha256=ok files=5 config.yaml=y kb=263168
base_warmup s0 model=y sha256=ok files=5 config.yaml=y kb=131768
base_warmup s1 model=y sha256=ok files=5 config.yaml=y kb=131768
base_warmup s2 model=y sha256=ok files=5 config.yaml=y kb=131768
```

- 当时的心跳：

```
2026-10-10T20:43:49Z waiting for queue 0040
2026-10-10T20:44:05Z done 0040
2026-10-10T20:43:54Z waiting for queue 0037
```
