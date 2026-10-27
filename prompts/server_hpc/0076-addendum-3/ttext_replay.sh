#!/usr/bin/env bash
# RUN-0076 补充 3（ITSC HPC hpcgpu06）：W2-c 加回放的定位与计数 T 文本读数（run_e1 --task refexp_box count --checkpoint <回放检查点>）。
#   B 段里这一步 rc=2：replay.sh 给 run_e1 的 --out 是 "$OUT/ttext"，目录名没有运行编号（D140 同一个错）。
#   代码用 wt0072（d38a5205；与 RUN-0069 的 7f53f240 之间 run_e1 / E1 / 适配器没有改动，只有 DEV-0063 给 E3 网格评测加的可选判据，
#   默认不变）。与 RUN-0069 在 06 上的 T 文本基座（0069_ttext/qwen3.5-9b）配对。
#   排队：等 $LOGS/eval_0076t.status 以 "done 0076t" 开头（补充 2 的 test 重跑跑完），再 claim_gpu。心跳 eval_0076x.status。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/ttext_replay.sh < /dev/null
# 彩排（开发机）：TINY=1 CODE=… REPO=… LOGS=… PY=… bash ttext_replay.sh（只核对目录名与命令行，不跑 run_e1：它的 --checkpoint 只用于正式运行）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0076x
M=qwen3.5-9b
WAIT_FILE=eval_0076t.status
WAIT_PREFIX="done 0076t"
if [ "$TINY" = 1 ]; then
  : "${CODE:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=tiny; CK=/nonexistent/tiny_ck; D=()
  export EV_NO_GIT=1
  OUT=$REPO/experiments/W2/0076_w2c_replay_ttext_tiny
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
  D=(--data-root "$VI/b200/data")
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
  OUT=$REPO/experiments/W2/0076_w2c_replay_ttext_hpcgpu06
fi
# run_e1 / run_w2 的规则：--out 的目录名必须以运行编号开头（CLAUDE.md 的"服务器驱动的写法"）
case "$(basename "$OUT")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "OUT 的目录名没有运行编号：$OUT" >&2; exit 2;; esac
CMD=("$PY" scripts/run_e1.py --model $M --task refexp_box count --subset eval ${D[@]+"${D[@]}"} \
     --checkpoint "$CK" --checkpoint-label t_text/w2c_replay/seed0 --out "$OUT")
if [ "$TINY" = 1 ]; then
  mkdir -p "$OUT"; printf '%q ' "${CMD[@]}" > "$OUT/cmd.txt"; echo "tiny：只核对目录名与命令行 → $OUT/cmd.txt"; exit 0
fi
# shellcheck disable=SC1091
source "$CODE/scripts/server/eval_lib.sh"
cd "$CODE" || exit 1
[ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
t0=$(date +%s)
while :; do
  st=$(cut -d' ' -f2- "$LOGS/$WAIT_FILE" 2>/dev/null || true)
  case "$st" in "$WAIT_PREFIX"*) break;; blocked*) progress "blocked: 补充 2 是 blocked：$st"; exit 1;; esac
  progress "waiting for $WAIT_FILE to start with '$WAIT_PREFIX' (now: ${st:-missing})"
  [ $(( $(date +%s) - t0 )) -ge 172800 ] && { progress "blocked: 等补充 2 超时"; exit 1; }
  sleep 60
done
NOK=0
claim_gpu "0076-addendum-3.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start code $(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE checkpoint manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
step ttext_w2c_replay "${CMD[@]}" || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "W2-c replay T-text ${RUN_ID} done (failed steps: $NOK)"
