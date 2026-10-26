#!/usr/bin/env bash
# RUN-0078（ITSC HPC hpcgpu06）：外部基线 LLMSeg，官方代码配本地 Qwen3.5-9B 零样本（W2 预注册 §13、§13a、§13b；D125-4、D135）。
#   1. llmseg：run_llmseg.py --split all，5 个 test 划分各 300 句（configs/w2_llmseg_subsets.json），主配置 T = 8；修正前（seg）与修正后（rec）两组读数。
#   2. compare：compare_llmseg.py，与 RUN-0072 的 W2-c 在同样 1,500 句上配对（同节点 06）。
# 排队：先等 WAIT_DONE 里每个心跳文件都以 done 开头（前一个任务跑完），再 claim_gpu（GPU 上没有进程）。默认等 eval_0072r.status（W2-c 补跑）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… EV_CAP_H=… [WAIT_DONE=…] bash …/llmseg.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash llmseg.sh（tiny 只跑 llmseg 一步，不比较）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0078
M=qwen3.5-9b
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=tiny; M=qwen3.5-0.8b
  T=(--tiny --split all); D=()
  WAIT_DONE=${WAIT_DONE:-}
  export EV_NO_GIT=1
else
  NODE=hpcgpu06
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  T=(--split all ${LLMSEG_ARGS:-}); D=(--data-root "$VI/b200/data")
  WAIT_DONE=${WAIT_DONE:-eval_0072r.status}
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  WAIT_MAX_S=${WAIT_MAX_S:-172800}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W2/${RUN_ID}_llmseg_$NODE
W2C_LEAF=$REPO/experiments/W2/0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
# 1) 等前一个任务跑完（心跳以 done 开头）
t0=$(date +%s)
for f in $WAIT_DONE; do
  while :; do
    st=$(cat "$LOGS/$f" 2>/dev/null || true)
    case "$(printf '%s' "$st" | cut -d' ' -f2-)" in done*) break;; esac
    case "$(printf '%s' "$st" | cut -d' ' -f2-)" in blocked*) progress "blocked: 前一个任务 $f 是 blocked，不开跑"; exit 1;; esac
    progress "waiting for $f to be done (now: ${st:-missing})"
    [ $(( $(date +%s) - t0 )) -ge "${WAIT_MAX_S:-172800}" ] && { progress "blocked: 等 $f 超时"; exit 1; }
    sleep 60
  done
done
NOK=0
claim_gpu "0078-hpc-06-llmseg.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE model $M subsets $(sha "$WT/configs/w2_llmseg_subsets.json" | cut -d' ' -f1) provenance $(sha "$WT/third_party/llmseg/PROVENANCE.md" | cut -d' ' -f1)" >> "$OUT/steps.log"
MEMPID=""
if [ "$TINY" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 1000 >> "$OUT/gpu_mem.csv" 2>/dev/null &
  MEMPID=$!
fi
RES=(); [ -d "$OUT/$M/refseg/llmseg" ] && RES=(--resume)
step llmseg "$PY" scripts/run_llmseg.py --model $M "${T[@]}" ${D[@]+"${D[@]}"} --out "$OUT" ${RES[@]+"${RES[@]}"} || NOK=$((NOK + 1))
if [ "$TINY" != 1 ]; then
  step compare "$PY" scripts/compare_llmseg.py --llmseg "$OUT/$M/refseg/llmseg" --w2c "$W2C_LEAF" --out "$OUT/compare_w2c" || NOK=$((NOK + 1))
fi
[ -n "$MEMPID" ] && kill "$MEMPID" 2>/dev/null
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "LLMSeg ${RUN_ID} done (failed steps: $NOK)"
