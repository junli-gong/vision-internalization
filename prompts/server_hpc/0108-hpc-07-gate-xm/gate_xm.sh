#!/usr/bin/env bash
# RUN-0108（ITSC HPC，hpcgpu07，只用 CPU）：X-M 的去留判定 G-XM（D165 第五节；预注册 XM-5、XM-5a（D170）、XM-5b；D176 的叶目录选法）。
#   输入都在 07 上：判定集 = RUN-0106 的 0106_xm_dec_r2（X-M）与 0106_base_dec（基座）；指代分割 = RUN-0107（X-M）对 RUN-0080（X-2 seed 0）；
#   大题量 = RUN-0107 的 0107_lb_xm 对 RUN-0084 的 lb_base（第 4 条，D170）与 lb_x2（D165 原文的诊断）。不调任何参数；判一次。
#   先逐个核对输入目录与文件都在，再跑 scripts/gate_xm.py --gate G-XM（它自己的核对不过就退出码 2、什么都不写）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。不占 GPU。
# 用法：srun --jobid=157297 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=… bash …/gate_xm.sh < /dev/null
# 彩排（开发机，TINY=1 REPO=<本地仓库>）：只核对输入的布局（git 里没有、列在 LARGE_FILES.manifest.tsv 里的文件算"只在 HPC 上"），不跑判定。
set -uo pipefail
TINY=${TINY:-0}
RUN_ID=0108
M=qwen3.5-9b
SP5=(explore_val_refcoco_unc_val "explore_val_refcoco+_unc_val" explore_val_refcocog_umd_val devtune800 devcheck2)
if [ "$TINY" = 1 ]; then
  : "${REPO:?本地仓库}"
  WT=$REPO; NODE=tiny
else
  NODE=hpcgpu07
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt$RUN_ID
  REPO=$VI/repo
  LOGS=$VI/logs
  PY=$VI/env/bin/python
  DATA=$VI/data   # X-M 的数据根（COCO 2017；load_pairs 读 xm_decision 的全部对）
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data MPLCONFIGDIR=$VI/cache/$NODE/mpl
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
  export EV_NO_GIT=1 EV_DF_PATH=$VI
  EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
fi
E=$REPO/experiments
CK=$E/XM/0104_xm_hpcgpu07/0104_xm_train/w2c/$M/seed0/ckpt/final/model
XMD=$E/XM/0106_xm_decision_hpcgpu07
XM_MAIN=$XMD/0106_xm_dec_r2/$M/xm/xm_spline_trimap/xm_decision
XM_BOX=$XMD/0106_xm_dec_r2/$M/xm/xm_box_grabcut/xm_decision
BASE_DET=$XMD/0106_base_dec/$M/xm/base_det/xm_decision
XMR=(); X2R=()
for s in "${SP5[@]}"; do
  XMR+=("$E/XM/0107_xm_refseg_lb_hpcgpu07/0107_refseg/$M/refseg/x2_spline_trimap/$s")
  X2R+=("$E/X/0080_x2_hpcgpu07/0080_eval/$M/refseg/x2_spline_trimap/$s")
done
LB_XM=$E/XM/0107_xm_refseg_lb_hpcgpu07/0107_lb_xm
LB_BASE=$E/X/0084_lb_explore_hpcgpu07/lb_base
LB_X2=$E/X/0084_lb_explore_hpcgpu07/lb_x2
OUT=$E/XM/${RUN_ID}_gate_xm_$NODE
GO=$OUT/${RUN_ID}_gate
for o in "$OUT" "$GO"; do case "$(basename "$o")" in [0-9][0-9][0-9][0-9]_*) ;; *) echo "目录名没有运行编号：$o" >&2; exit 2;; esac; done
# 输入布局：目录与必需的文件。TINY=1 时 git 里没有的文件，要列在所在运行目录的 LARGE_FILES.manifest.tsv 里（只在 HPC 上）。
miss=0
need() {  # need <文件>
  [ -f "$1" ] && return 0
  if [ "$TINY" = 1 ]; then
    local r m rel
    for r in "$XMD" "$E/XM/0107_xm_refseg_lb_hpcgpu07" "$E/X/0080_x2_hpcgpu07" "$E/X/0084_lb_explore_hpcgpu07"; do
      case "$1" in "$r"/*) m=$r/LARGE_FILES.manifest.tsv; rel=${1#"$r"/}
        [ -f "$m" ] && cut -f1 "$m" | grep -qxF "$rel" && { echo "只在 HPC 上：${1#"$REPO"/}"; return 0; };; esac
    done
  fi
  echo "缺：$1" >&2; miss=$((miss + 1))
}
[ "$TINY" = 1 ] || { need "$CK/manifest.json"; [ -d "$DATA" ] || { echo "缺数据根：$DATA" >&2; miss=$((miss + 1)); }; }
for d in "$XM_MAIN" "$XM_BOX" "$BASE_DET" "${XMR[@]}" "${X2R[@]}"; do
  for f in predictions.jsonl metrics.json config.yaml meta.json; do need "$d/$f"; done
done
for d in "$LB_XM" "$LB_BASE" "$LB_X2"; do for f in lang_battery.json predictions.jsonl config.yaml meta.json; do need "$d/$f"; done; done
[ "$miss" = 0 ] || { echo "输入缺 $miss 个文件，不判定" >&2; [ "$TINY" = 1 ] && exit 1; }
if [ "$TINY" = 1 ]; then echo "布局核对通过（TINY=1 不跑判定）"; exit 0; fi
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "$miss" = 0 ] || { progress "blocked: 输入缺 $miss 个文件"; exit 1; }
[ -e "$GO/gate_xm.json" ] && { progress "blocked: $GO/gate_xm.json 已存在（只判一次）"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE x-m checkpoint manifest $(sha "$CK/manifest.json" | cut -d' ' -f1)" >> "$OUT/steps.log"
step gate_xm "$PY" scripts/gate_xm.py --gate G-XM --data-root "$DATA" --xm-checkpoint "$CK" \
  --xm-main "$XM_MAIN" --xm-box-grabcut "$XM_BOX" --base-det "$BASE_DET" \
  --xm-refseg "${XMR[@]}" --x2-refseg "${X2R[@]}" \
  --lb-xm "$LB_XM" --lb-base "$LB_BASE" --lb-x2 "$LB_X2" --criterion4 d170 --out-dir "$GO"
RC=$?
echo "$(now) done; gate_xm rc=$RC" >> "$OUT/steps.log"
progress "done $RUN_ID (gate_xm rc=$RC)"
push_out "G-XM gate ${RUN_ID} done (gate_xm rc=$RC)"
