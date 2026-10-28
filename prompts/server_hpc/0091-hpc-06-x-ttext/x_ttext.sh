#!/usr/bin/env bash
# RUN-0091（ITSC HPC hpcgpu06）：探索候选 X-2（C1）、X-3（C2）的定位与计数 T 文本读数（run_e1 --task refexp_box count --checkpoint …；D147-4）。
#   给 9B 语言代价四方对照表（E3b+C3、W2-c、W2-c+回放、X-2）补定位与计数两行；X-3 一并读，作消融。
#   与 RUN-0069 在 06 上的 T 文本基座（0069_ttext/qwen3.5-9b）配对：从 7f53f240 到现在，run_e1 / E1 / 适配器没有改动
#   （只有 DEV-0063 给 E3 网格评测加的可选判据，默认不变；D141 核对过）。06 空闲，不等前一个任务。心跳 eval_0091.status。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/x_ttext.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash x_ttext.sh（只核对目录名与命令行）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0091
M=qwen3.5-9b
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=tiny; D=(); X2_CK=/nonexistent/x2; X3_CK=/nonexistent/x3
  export EV_NO_GIT=1
  OUT=$REPO/experiments/X/0091_x_ttext_tiny
else
  NODE=hpcgpu06
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0091
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  X2_CK=$REPO/experiments/X/0080_x2_hpcgpu07/0080_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
  X3_CK=$REPO/experiments/X/0082_x3_hpcgpu07/0082_train/w2c/qwen3.5-9b/seed0/ckpt/final/model
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
  OUT=$REPO/experiments/X/0091_x_ttext_hpcgpu06
fi
case "$(basename "$OUT")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "OUT 的目录名没有运行编号：$OUT" >&2; exit 2;; esac
cmd() { echo "$PY" scripts/run_e1.py --model $M --task refexp_box count --subset eval ${D[@]+"${D[@]}"} --checkpoint "$2" --checkpoint-label "t_text/$1/seed0" --out "$OUT"; }
if [ "$TINY" = 1 ]; then
  mkdir -p "$OUT"; { cmd x2 "$X2_CK"; cmd x3 "$X3_CK"; } > "$OUT/cmd.txt"; echo "tiny：只核对目录名与命令行 → $OUT/cmd.txt"; exit 0
fi
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
NOK=0
claim_gpu "0091-hpc-06-x-ttext.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"
for pair in "x2:$X2_CK" "x3:$X3_CK"; do
  n=${pair%%:*}; ck=${pair#*:}
  [ -f "$ck/manifest.json" ] || { echo "$(now) 没有检查点 $ck" >> "$OUT/steps.log"; NOK=$((NOK + 1)); continue; }
  echo "$(now) $n checkpoint manifest $(sha "$ck/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
  step "ttext_$n" "$PY" scripts/run_e1.py --model $M --task refexp_box count --subset eval ${D[@]+"${D[@]}"} \
    --checkpoint "$ck" --checkpoint-label "t_text/$n/seed0" --out "$OUT" || NOK=$((NOK + 1))
done
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "X-2/X-3 T-text ${RUN_ID} done (failed steps: $NOK)"
