#!/usr/bin/env bash
# RUN-0084（ITSC HPC）：探索候选的大题量语言读数（scripts/lang_battery.py；DEV-0064；D145）。只描述，用于选候选与论文；判定仍按健康检查（预注册 X §3）。
#   NODE=hpcgpu07：基座、X-2（RUN-0080）、X-3（RUN-0082）三个，同节点配对，再 --compare 基座对另两个。
#   NODE=hpcgpu06：X-3L（RUN-0083），与 RUN-0074 在 06 上的 lb_base 配对；排在 RUN-0083 之后（等 eval_0083.status 以 "done 0083" 开头）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=… --overlap -n1 -c 8 env PATH=… TINY=0 NODE=… EV_CAP_H=… bash …/lb_explore.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… RAW=… bash lb_explore.sh（只跑基座与基座对基座的 --compare）
set -uo pipefail
TINY=${TINY:-0}
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${RAW:?}"
  NODE=tiny; RUN_ID=0084-tiny; M=qwen3.5-0.8b; T=(--tiny --limit 3); D=()
  export NLTK_DATA=$WT/tests/fixtures/health/nltk_data EV_NO_GIT=1
else
  : "${NODE:?hpcgpu07 或 hpcgpu06}"
  RUN_ID=0084-$NODE  # 心跳 eval_0084-<节点>.status：两个节点同时跑，各用各的
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0084
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  RAW=$VI/data/lang_battery_raw
  M=qwen3.5-9b
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data")
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/X/0084_lb_explore_$NODE
case "$(basename "$OUT")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "OUT 的目录名没有运行编号：$OUT" >&2; exit 2;; esac
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$NODE" = hpcgpu06 ]; then
  t0=$(date +%s)
  while :; do
    st=$(cut -d' ' -f2- "$LOGS/eval_0083.status" 2>/dev/null || true)
    case "$st" in "done 0083"*) break;; blocked*) progress "blocked: RUN-0083 是 blocked：$st"; exit 1;; esac
    progress "waiting for eval_0083.status to start with 'done 0083' (now: ${st:-missing})"
    [ $(( $(date +%s) - t0 )) -ge 172800 ] && { progress "blocked: 等 RUN-0083 超时"; exit 1; }
    sleep 60
  done
fi
NOK=0
claim_gpu "0084-hpc-lb-explore.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"
LB=(--model "$M" ${T[@]+"${T[@]}"} --raw-dir "$RAW")
run_ck() {  # 名字 检查点
  local name=$1 ck=$2
  [ -f "$ck/manifest.json" ] || { echo "$(now) 没有检查点 $ck" >> "$OUT/steps.log"; NOK=$((NOK + 1)); return; }
  echo "$(now) $name checkpoint manifest $(sha "$ck/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
  step "lb_$name" "$PY" scripts/lang_battery.py "${LB[@]}" --label "$name" ${D[@]+"${D[@]}"} --checkpoint "$ck" --out "$OUT/lb_$name" || NOK=$((NOK + 1))
}
X=$REPO/experiments/X
case "$NODE" in
  tiny)
    step lb_base "$PY" scripts/lang_battery.py "${LB[@]}" --label base --out "$OUT/lb_base" || NOK=$((NOK + 1))
    step lb_compare_self "$PY" scripts/lang_battery.py --compare "$OUT/lb_base" "$OUT/lb_base" --out "$OUT/lb_compare_self.json" || NOK=$((NOK + 1));;
  hpcgpu07)
    step lb_base "$PY" scripts/lang_battery.py "${LB[@]}" --label base --out "$OUT/lb_base" || NOK=$((NOK + 1))
    run_ck x2 "$X/0080_x2_hpcgpu07/0080_train/w2c/$M/seed0/ckpt/final/model"
    run_ck x3 "$X/0082_x3_hpcgpu07/0082_train/w2c/$M/seed0/ckpt/final/model"
    for n in x2 x3; do
      step "lb_compare_$n" "$PY" scripts/lang_battery.py --compare "$OUT/lb_base" "$OUT/lb_$n" --out "$OUT/lb_compare_$n.json" || NOK=$((NOK + 1))
    done;;
  hpcgpu06)
    run_ck x3l "$X/0083_x3l_hpcgpu06/0083_train/w2c/$M/seed0/ckpt/final/model"
    step lb_compare_x3l "$PY" scripts/lang_battery.py --compare "$REPO/experiments/W2/0074_lang_battery_hpcgpu06/lb_base" "$OUT/lb_x3l" \
      --out "$OUT/lb_compare_x3l.json" || NOK=$((NOK + 1));;
esac
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "lang battery explore ${RUN_ID} $NODE done (failed steps: $NOK)"
