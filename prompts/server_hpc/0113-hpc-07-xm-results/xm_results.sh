#!/usr/bin/env bash
# RUN-0113（ITSC HPC hpcgpu07，只用 CPU）：X-M 结果汇总的配对 CI（scripts/mgr/xm_results_table.py；管理员的分析脚本，只读结果目录）。
#   代码在稀疏的工作树 wt0113，结果在 $REPO（--root $REPO）；输出 experiments/XM/0113_xm_results_hpcgpu07/xm_results.{json,md}。
#   不占 GPU，不写暂停文件。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
# 用法：srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/xm_results.sh < /dev/null
# 彩排（开发机）：TINY=1 WT=<本仓库> REPO=<含 experiments/ 的目录> LOGS=… PY=… bash xm_results.sh
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0113
if [ "$TINY" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${LOGS:?}" "${PY:?}"
  NODE=tiny
  export EV_NO_GIT=1
else
  NODE=hpcgpu07
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=$VI/cache/$NODE/mpl
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export EV_NO_GIT=1 EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
OUT=$REPO/experiments/XM/${RUN_ID}_xm_results_$NODE
case "$(basename "$OUT")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$OUT" >&2; exit 2;; esac
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE root $REPO" >> "$OUT/steps.log"
step xm_results "$PY" scripts/mgr/xm_results_table.py --root "$REPO" --out "experiments/XM/${RUN_ID}_xm_results_$NODE/xm_results"
RC=$?
echo "$(now) done; xm_results rc=$RC" >> "$OUT/steps.log"
progress "done $RUN_ID (xm_results rc=$RC)"
push_out "X-M results table ${RUN_ID} done (rc=$RC)"
