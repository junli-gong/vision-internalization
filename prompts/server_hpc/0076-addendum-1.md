[RUN-TASK 0076 补充 1]（2026-10-26 04:3xZ）：启动命令加 `TINY=0`

- **现象**（服务器 CC 报告）：A 段 data 第一次启动立即退出，`replay.sh: line 52: TINY: unbound variable`。没有建结果目录，没有写暂停文件，GPU 上没有进程。
- **原因**：驱动在 `set -u` 下，第 52 行（`TRAIN_OUT` 的 tiny 分支）在 source `eval_lib.sh` 之前读 `$TINY`，而 `TINY` 的缺省值是 `eval_lib.sh` 里设的。开发机彩排给了 `TINY=1`，没有走到这条路径。
- **处理**：每次启动 replay.sh（A 段 data、A 段 train、B 段 eval），srun 的 env 里都加 `TINY=0`。代码不改，15 个 sha256 不变。
- **教训**：用 `set -u` 的驱动，`TINY` 要在文件开头就给缺省值。以后的驱动照此写，并在彩排里另跑一次不设 `TINY` 的语法路径。
