#!/usr/bin/env bash
# 开发机按服务器的内核绑定口径跑全量测试（DEV-0003-a1）。
#
# 服务器装了 flash-linear-attention（fla）：transformers 在 import 时把 Qwen3.5 的两个 GDN 模块级函数绑到 fla 的
# Triton 内核，只收 GPU 张量。开发机没装，走 torch，所以同一批 CPU 测试会"开发机绿、服务器红"（RUN-0003：16 个）。
# 本脚本把 tests/_fakes 放在 PYTHONPATH 最前面，让 transformers 在开发机上也绑到一个只收 GPU 张量的假 fla
# （说明见 tests/_fakes/fla/__init__.py），然后跑全量测试；子进程（冒烟脚本的 --tiny 彩排）继承同一个 PYTHONPATH。
# 同时设 VISINT_FAKE_FLA=1：测试据此核对这个口径确实生效。普通的 pytest 不设它，测试反过来核对假包不可见。
#
# 用法（开发机）：
#   conda run -n visint bash scripts/pytest_server_parity.sh -q
# 其余参数原样传给 pytest。解释器取 PATH 上的 python，可用环境变量 PYTHON 另指。
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo}/tests/_fakes${PYTHONPATH:+:${PYTHONPATH}}"
export VISINT_FAKE_FLA=1
cd "${repo}"
exec "${PYTHON:-python}" -m pytest tests/ "$@"
