#!/usr/bin/env bash
# RUN-0076 补充 2（ITSC HPC hpcgpu06）：W2-c 加回放在 5 个 test 划分上的评测（run_w2 --setting w2c --checkpoint <回放检查点>）。
#   B 段里这一步全部 rc=2：replay.sh 给 run_w2 的 --out 是 "$OUT/test"，目录名没有运行编号（run_w2 的规则；tiny 彩排不查这条）。
#   代码用 RUN-0072 的工作树 wt0072（与 W2-c 的 test 同一个 commit），结果写到新目录，与 RUN-0072 的 W2-c 同节点配对。
#   排队：等 $LOGS/eval_0076.status 以 "done 0076 [eval]" 开头（B 段其余步骤跑完），再 claim_gpu。心跳 eval_0076t.status。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/test_replay.sh < /dev/null
# 彩排（开发机）：TINY=1 CODE=… REPO=… LOGS=… PY=… CK=<tiny 检查点> bash test_replay.sh
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0076t
M=qwen3.5-9b
SPLITS=(test_refcoco_unc_testA test_refcoco_unc_testB "test_refcoco+_unc_testA" "test_refcoco+_unc_testB" test_refcocog_umd_test)
WAIT_FILE=eval_0076.status
WAIT_PREFIX="done 0076 [eval]"
if [ "$TINY" = 1 ]; then
  : "${CODE:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}" "${CK:?}"
  NODE=tiny; M=qwen3.5-0.8b; SPLITS=(devtune800)
  T=(--tiny); D=(); CW=1
  export EV_NO_GIT=1
  OUT=$REPO/experiments/W2/0076_w2c_replay_test_tiny
else
  NODE=hpcgpu06
  VI=/home/mail/chenyuma/visint
  CODE=$VI/wt/wt0072
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  CK=$REPO/experiments/W2/0076_w2c_replay_hpcgpu07/0076_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(); D=(--data-root "$VI/b200/data"); CW=8
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
  OUT=$REPO/experiments/W2/0076_w2c_replay_test_hpcgpu06
fi
# run_w2 的规则：--out 的目录名必须以运行编号开头（RUN-0061、0076 B 段两次因此失败；tiny 不查，这里自己查）
case "$(basename "$OUT")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "OUT 的目录名没有运行编号：$OUT" >&2; exit 2;; esac
# shellcheck disable=SC1091
source "$CODE/scripts/server/eval_lib.sh"
cd "$CODE" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$TINY" != 1 ] && ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('configs/w2.json'))['frozen'] is True else 1)"; then
  progress "blocked: configs/w2.json 没有冻结"; exit 1
fi
if [ "$TINY" != 1 ]; then
  t0=$(date +%s)
  while :; do
    st=$(cut -d' ' -f2- "$LOGS/$WAIT_FILE" 2>/dev/null || true)
    case "$st" in "$WAIT_PREFIX"*) break;; blocked*) progress "blocked: B 段是 blocked：$st"; exit 1;; esac
    progress "waiting for $WAIT_FILE to start with '$WAIT_PREFIX' (now: ${st:-missing})"
    [ $(( $(date +%s) - t0 )) -ge 172800 ] && { progress "blocked: 等 B 段超时"; exit 1; }
    sleep 60
  done
fi
NOK=0
claim_gpu "0076-addendum-2.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start code $(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE checkpoint manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
for SP in "${SPLITS[@]}"; do
  step "w2c_replay_$SP" "$PY" scripts/run_w2.py --model "$M" ${T[@]+"${T[@]}"} ${D[@]+"${D[@]}"} --split "$SP" --w2-config configs/w2.json \
    --out "$OUT" --candidate-workers "$CW" --task refseg --setting w2c --checkpoint "$CK" || NOK=$((NOK + 1))
done
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W2-c replay test ${RUN_ID} done (failed steps: $NOK)"
