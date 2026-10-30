#!/usr/bin/env bash
# RUN-0097（ITSC HPC hpcgpu06）：外部对照 LISA-7B 在 W2 test 的 1,500 句子集上（D154、D156；预注册 U-2；DEV-0075）。
#   前提（任务书第 1–2 步，登录节点上手工做完）：$VI/lisa/venv 建好并装了 requirements_lisa.txt；权重下到 $VI/hf/hub 并 --verify 通过。
#   驱动：claim_gpu → 导出清单（我们的环境）→ LISA 试跑 20 句/划分（LISA 的 venv）→ 续跑全量 → 计分（我们的环境）。
#   隔离（硬约束第 1 条）：LISA 的输出只在 experiments/W2/0097_lisa_hpcgpu06，只用来计分出表。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/lisa.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=… REPO=… LOGS=… PY=… bash lisa.sh（tests/fixtures 的小数据 + 替身运行器，不加载权重）
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0097
LISA_REV=43c754eef75871fcc11c84d3930a402f7b0a754f
CLIP_REV=32bd64288804d66eefd0ccbe215aa642df71cc41
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=tiny; LPY=$PY
  EXP=(--tiny); RUNX=(--runner-factory tests._lisa_fakes:make_fixed); SCX=(--allow-stand-in); D=()
  export EV_NO_GIT=1
else
  NODE=hpcgpu06
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  LPY=$VI/lisa/venv/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  LW=$HF_HUB_CACHE/models--xinlai--LISA-7B-v1/snapshots/$LISA_REV
  CW=$HF_HUB_CACHE/models--openai--clip-vit-large-patch14/snapshots/$CLIP_REV
  EXP=(--data-root "$VI/b200/data"); RUNX=(--lisa-weights "$LW" --clip-weights "$CW"); SCX=(); D=(--data-root "$VI/b200/data")
  export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1
  export EV_DF_PATH=$VI
  CLAIM_MAX_S=${CLAIM_MAX_S:-3600}
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/W2/${RUN_ID}_lisa_$NODE
case "$(basename "$OUT")" in [0-9][0-9][0-9][0-9]_lisa*) ;; *) echo "目录名没有运行编号或不含 _lisa：$OUT" >&2; exit 2;; esac
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
if [ "$TINY" != 1 ]; then
  [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
  [ -x "$LPY" ] || { progress "blocked: 没有 LISA 的 venv（$LPY）"; exit 1; }
  [ -d "$LW" ] && [ -d "$CW" ] || { progress "blocked: 权重快照不在（$LW / $CW）"; exit 1; }
  "$PY" scripts/lisa_w2.py weights --verify > "$LOGS/lisa_0097_verify.log" 2>&1 || { progress "blocked: 权重核对失败（见 lisa_0097_verify.log）"; exit 1; }
fi
NOK=0
claim_gpu "0097-hpc-06-lisa.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
mkdir -p "$OUT"
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE lisa $LISA_REV clip $CLIP_REV" >> "$OUT/steps.log"
die() { echo "$(now) ABORT: $*" | tee -a "$OUT/steps.log"; progress "blocked: $*"; push_out "LISA ${RUN_ID} aborted: $*"; exit 1; }
# 清单已导出（驱动重启时）就不再导出：src/lisa/common 的布局 <out>/lisa-7b-v1/inputs/manifest.json
[ -f "$OUT/lisa-7b-v1/inputs/manifest.json" ] || \
  step lisa_export "$PY" scripts/lisa_w2.py export --out "$OUT" ${EXP[@]+"${EXP[@]}"} --test-approval D154 || die "导出清单失败"
# 驱动重启时 raw 已有输出：试跑也加 --resume（已写出的前缀跳过，身份必须相同）
RES=(); [ -n "$(ls -A "$OUT/lisa-7b-v1/raw" 2>/dev/null)" ] && RES=(--resume)
step lisa_smoke "$LPY" scripts/run_lisa.py --out "$OUT" --test-approval D154 ${RUNX[@]+"${RUNX[@]}"} --limit 20 ${RES[@]+"${RES[@]}"} || die "试跑失败"
step lisa_full "$LPY" scripts/run_lisa.py --out "$OUT" --test-approval D154 ${RUNX[@]+"${RUNX[@]}"} --resume || NOK=$((NOK + 1))
SRES=(); [ -d "$OUT/lisa-7b-v1/refseg/lisa_7b" ] && SRES=(--rescore)  # 驱动重启时叶目录已在：重写
step lisa_score "$PY" scripts/lisa_w2.py score --out "$OUT" ${D[@]+"${D[@]}"} ${SCX[@]+"${SCX[@]}"} ${SRES[@]+"${SRES[@]}"} || NOK=$((NOK + 1))
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "LISA ${RUN_ID} done (failed steps: $NOK)"
