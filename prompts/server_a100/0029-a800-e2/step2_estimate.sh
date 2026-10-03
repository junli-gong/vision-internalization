#!/usr/bin/env bash
# RUN-0029 第 2 步（GPU）：Qwen 四个尺寸的 E2 估算（四个任务 + 两条参考线），再按批准的上限判定 GO / STOP。约 20 分钟。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
cd "$CODE" || exit 1
EST="$OUT/0029_e2_estimate"
for m in "${QWEN[@]}"; do
  [ "$TINY" = 1 ] && [ "$m" != qwen3.5-0.8b ] && continue   # 彩排只有 0.8B 的小模型
  python scripts/run_e2.py --estimate --model "$m" --task all "${REFS[@]}" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} ${TFLAG[@]+"${TFLAG[@]}"} \
    --out "$EST" > "$OUT/e2_estimate_$m.log" 2>&1
  step "e2_estimate_$m" $?
done
MODELS=("${QWEN[@]}"); [ "$TINY" = 1 ] && MODELS=(qwen3.5-0.8b)
python scripts/mgr/e2_budget_check.py "$EST" --cap "$CAP" --models "${MODELS[@]}" --out "$OUT/budget_check.json" > "$OUT/budget_check.log" 2>&1
rc=$?; step budget_check $rc
grep -E '"(total_gpu_hours|decision)"' "$OUT/budget_check.json"
if [ $rc -eq 0 ]; then echo "[step2] GO。下一步：bash step3_run.sh ${WORK}（约一天，放进 tmux）"
else echo "[step2] STOP：不要运行第 3 步。直接 bash step4_collect.sh $WORK 回传。"; fi
exit $rc
