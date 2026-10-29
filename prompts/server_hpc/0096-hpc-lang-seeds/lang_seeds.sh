#!/usr/bin/env bash
# RUN-0096（ITSC HPC）：主方法 C1 的 seed 1、seed 2 的语言读数（D153）。只描述，用来看语言代价换种子后是否站得住。
#   NODE=hpcgpu07（07 空闲，现在就跑）：seed 1（RUN-0093）的大题量（lang_battery.py），与 RUN-0084 在 07 上的 lb_base 配对。
#   NODE=hpcgpu06（排在 RUN-0094 之后，等 eval_0094.status 以 "done 0094" 开头）：
#     seed 1、seed 2 的 T 文本定位与计数（run_e1 --task refexp_box count），与 RUN-0069 在 06 上的 T 文本基座配对（同 RUN-0091）；
#     seed 2（RUN-0094）的大题量，与 RUN-0074 在 06 上的 lb_base 配对（同 RUN-0084 的 X-3L）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=<157297 或 157296> --overlap -n1 -c 8 env PATH=… TINY=0 NODE=… EV_CAP_H=… bash …/lang_seeds.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash lang_seeds.sh（只核对目录名与两个节点的命令行）
set -uo pipefail
TINY=${TINY:-0}
M=qwen3.5-9b
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  D=(); RAW=/nonexistent/raw
  export EV_NO_GIT=1
else
  : "${NODE:?hpcgpu07 或 hpcgpu06}"
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0096
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  RAW=$VI/data/lang_battery_raw
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
fi
X=$REPO/experiments/X
S1_CK=$X/0093_x2s1_hpcgpu07/0093_train/w2c/$M/seed1/ckpt/final/model
S2_CK=$X/0094_x2s2_hpcgpu06/0094_train/w2c/$M/seed2/ckpt/final/model
LB07_BASE=$X/0084_lb_explore_hpcgpu07/lb_base
LB06_BASE=$REPO/experiments/W2/0074_lang_battery_hpcgpu06/lb_base
out_of() { echo "$REPO/experiments/X/0096_lang_seeds_$1"; }
for n in hpcgpu07 hpcgpu06 tiny; do
  case "$(basename "$(out_of $n)")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$(out_of $n)" >&2; exit 2;; esac
done
lb_cmd() { echo "$PY" scripts/lang_battery.py --model "$M" --raw-dir "$RAW" --label "$1" ${D[@]+"${D[@]}"} --checkpoint "$2" --out "$3/lb_$1"; }
lbc_cmd() { echo "$PY" scripts/lang_battery.py --compare "$1" "$2/lb_$3" --out "$2/lb_compare_$3.json"; }
tt_cmd() { echo "$PY" scripts/run_e1.py --model $M --task refexp_box count --subset eval ${D[@]+"${D[@]}"} --checkpoint "$2" --checkpoint-label "t_text/$1/$4" --out "$3"; }
if [ "$TINY" = 1 ]; then
  O=$(out_of tiny); mkdir -p "$O"
  { echo "# hpcgpu07"; O7=$(out_of hpcgpu07); lb_cmd x2s1 "$S1_CK" "$O7"; lbc_cmd "$LB07_BASE" "$O7" x2s1
    echo "# hpcgpu06"; O6=$(out_of hpcgpu06); tt_cmd x2s1 "$S1_CK" "$O6" seed1; tt_cmd x2s2 "$S2_CK" "$O6" seed2
    lb_cmd x2s2 "$S2_CK" "$O6"; lbc_cmd "$LB06_BASE" "$O6" x2s2; } > "$O/cmd.txt"
  echo "tiny：只核对目录名与命令行 → $O/cmd.txt"; exit 0
fi
case "$NODE" in hpcgpu07|hpcgpu06) ;; *) echo "NODE 只能是 hpcgpu07 或 hpcgpu06：$NODE" >&2; exit 2;; esac
RUN_ID=0096-$NODE  # 心跳 eval_0096-<节点>.status
OUT=$(out_of "$NODE")
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
if [ "$NODE" = hpcgpu06 ]; then
  t0=$(date +%s)
  while :; do
    st=$(cut -d' ' -f2- "$LOGS/eval_0094.status" 2>/dev/null || true)
    case "$st" in "done 0094"*) break;; blocked*) progress "blocked: RUN-0094 是 blocked：$st"; exit 1;; esac
    progress "waiting for eval_0094.status to start with 'done 0094' (now: ${st:-missing})"
    [ $(( $(date +%s) - t0 )) -ge 172800 ] && { progress "blocked: 等 RUN-0094 超时"; exit 1; }
    sleep 60
  done
fi
NOK=0
claim_gpu "0096-hpc-lang-seeds.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"
have() {  # 名字 检查点
  [ -f "$2/manifest.json" ] || { echo "$(now) 没有检查点 $2" >> "$OUT/steps.log"; NOK=$((NOK + 1)); return 1; }
  echo "$(now) $1 checkpoint manifest $(sha "$2/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
}
lb() {  # 名字 检查点 基座
  have "$1" "$2" || return
  step "lb_$1" $(lb_cmd "$1" "$2" "$OUT") || { NOK=$((NOK + 1)); return; }
  step "lb_compare_$1" $(lbc_cmd "$3" "$OUT" "$1") || NOK=$((NOK + 1))
}
tt() {  # 名字 检查点 种子标签
  have "$1" "$2" || return
  step "ttext_$1" $(tt_cmd "$1" "$2" "$OUT" "$3") || NOK=$((NOK + 1))
}
case "$NODE" in
  hpcgpu07) lb x2s1 "$S1_CK" "$LB07_BASE";;
  hpcgpu06) tt x2s1 "$S1_CK" seed1; tt x2s2 "$S2_CK" seed2; lb x2s2 "$S2_CK" "$LB06_BASE";;
esac
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "C1 seeds language ${RUN_ID} done (failed steps: $NOK)"
