#!/usr/bin/env bash
# RUN-0035 第 3 步（A800 约 95 小时）：4B 全部变体 × 3 个种子（D77-4：A800 训 0.8B 与 4B）。中断了重新运行同一条命令。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
cd "$CODE" || exit 1
M=qwen3.5-4b; [ "$TINY" = 1 ] && M=qwen3.5-0.8b
for s in "${SEEDS[@]}"; do
  size_seed "$M" "$s" 0
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) 4B seed $s done" >> "$OUT/milestones.txt"
  [ "$TINY" = 1 ] && break
done
progress "step3 done"
echo "[step3] 完成。请运行 bash step4_collect.sh $WORK 回传。"
