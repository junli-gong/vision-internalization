#!/usr/bin/env bash
# visint 停车开关（2026-10-10，RUN-0044）：存在 run/park-<节点> 就占着卡什么也不做；否则照旧启动 MiniMax。恢复：删开关、杀 sleep。
BASE="${MINIMAX_H3_BASE:-/home/mail/chenyuma/minimax-h3-fp8}"
if [ -e "$BASE/run/park-$(hostname -s)" ]; then
  echo "[$(date -u +%FT%TZ)] parked on $(hostname -s) by visint (run/park-$(hostname -s))" >&2
  exec sleep infinity
fi
exec "$BASE/start_minimax_h3_a100_1gpu_fp16.sh.orig" "$@"
