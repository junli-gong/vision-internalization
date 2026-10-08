#!/usr/bin/env bash
# RUN-0035 补充 1（D83，2026-10-08）：第 1 步之后，先只训 C3：0.8B 三个种子，再 4B 三个种子（A800 约 20 小时）。
# 原因：2B 的结果显示，只用网格或稠密监督训练会冲掉语言；负责人把 C3 与新的 E3b+C3 排到最前面，其余变体往后排。
# E3b+C3 的代码另发一个包。本步训完的 C3 之后照常算作第 2、3 步里的 C3：同名运行已完成就自动跳过，不会重训。
# 中断了重新运行同一条命令即可。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
cd "$CODE" || exit 1
R="$OUT/0035_e3_train"; C="$CKPT/train"
for m in qwen3.5-0.8b qwen3.5-4b; do
  mm=$m; [ "$TINY" = 1 ] && mm=qwen3.5-0.8b
  for s in "${SEEDS[@]}"; do
    train "c3_${m}_s${s}" scripts/train_e3b.py --model "$mm" --supervision text --seed "$s" --out "$R" --ckpt-root "$C"
    [ "$TINY" = 1 ] && break
  done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) C3 $m done" >> "$OUT/milestones.txt"
done
progress "step2c done"
echo "[step2c] 完成。请运行 bash step4_collect.sh $WORK 回传一次（C3 的结果与检查点）。E3b+C3 的新包到了再按它的说明继续；原来的 step2、step3 先不要跑。"
