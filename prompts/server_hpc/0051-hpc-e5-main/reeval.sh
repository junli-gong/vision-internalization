#!/usr/bin/env bash
# RUN-0051 补充 1（D96）：按判据 e5-v2 补评被 v1 拒绝的 E5 检查点；v1 下通过、已经评过 test 的只做核对（--check-only），不重评 test。
# 在主驱动暂停期间（$LOGS/train.pause.$NODE 存在）、在同一台节点上经 srun 跑（D93-1：同一组的评测都在训练它的节点上）。
# 跑完删掉主驱动的暂停文件，主驱动接着跑；这时工作树已切到 e5-v2 的 commit，之后的评测都用新判据。
# - 只处理本节点的运行：MAIN_OUT = 0051_e5_main_$NODE。评测结果写进主驱动的结果目录（同一个 --out，与 queue.sh 的 e5() 同一条命令），
#   本脚本的驱动记录放在 <MAIN_OUT>/reeval_e5v2/（done.txt、failed.txt、gpu_seconds、logs/），不占主驱动的用量上限。
# - 名单在开始时定，写进 <OUT>/plan.tsv：
#   补评（eval_<tag>）：主驱动 failed.txt 里 eval_* 的名字，不在主驱动 done.txt 里（D96-1：被 v1 拒绝的统一补评）；
#   只核对（mcheck_<tag>）：主驱动 done.txt 里 eval_* 的名字，且模型目录里 merge_check.json 的 criteria_version 不是 e5-v2。
# - 补评成功的名字同时追加进主驱动的 done.txt（主驱动以后重启也不会再评一次）；主驱动的 failed.txt 不动（历史记录）。
# - 最后写 <OUT>/summary.tsv：每个运行的 e5-v2 判定、v1 判定（只报告）、指标层面的差与标记（D96-2）。不含任何 test 分数。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法（在节点上经 srun 启动，见 0051-addendum-1.md）：
#   srun --jobid=<任务> --overlap -n1 env NODE=hpcgpu07 EXPECT_COMMIT=<40 位> PATH=… bash $VI/wt/wt0051/prompts/server_hpc/0051-hpc-e5-main/reeval.sh < /dev/null
# 彩排（开发机）：TINY=1 NODE=hpcgpu07 EXPECT_COMMIT=… WT=… REPO=… MAIN_OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… bash reeval.sh
NODE=${NODE:?NODE=hpcgpu06 或 hpcgpu07}
case "$NODE" in hpcgpu06|hpcgpu07) ;; *) echo "未知节点 $NODE" >&2; exit 2 ;; esac
EXPECT_COMMIT=${EXPECT_COMMIT:?任务书给的 commit}
RUN_ID=0051r-$NODE
M=qwen3.5-2b
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${MAIN_OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}"
  CHUNK=1; EXTRA=(--tiny); EVAL_LIMIT=(--limit 2)
  M=${TINY_MODEL:-qwen3.5-0.8b}  # 彩排可换更快的小模型
  export TQ_NO_GIT=1
else
  VI=/home/mail/chenyuma/visint
  WT=$VI/wt/wt0051
  REPO=$VI/repo
  LOGS=$VI/logs
  MAIN_OUT=$REPO/experiments/E5/0051_e5_main_$NODE
  CK_TMP=$VI/ckpt/$NODE/tmp0051
  CK_FINAL=$VI/ckpt/$NODE/0051
  PY=$VI/env/bin/python
  CHUNK=1
  export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
  export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  export TQ_NO_GIT=1
  export TQ_DF_PATH=$VI
  MEM_STOP_GB=100
  EXTRA=(); EVAL_LIMIT=()
  QUEUE_CAP_H=${QUEUE_CAP_H:?任务书的上限}
fi
OUT=$MAIN_OUT/reeval_e5v2
MAIN_DRV=$MAIN_OUT/driver
MAIN_PAUSE=$LOGS/train.pause.$NODE
export TQ_PAUSE=$LOGS/train.pause.$NODE.reeval  # 本脚本自己的暂停文件（主驱动的暂停文件在跑的期间一直存在，跑完由本脚本删掉）
export MEM_STOP_GB 2>/dev/null
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || die "不在 srun 的资源范围里（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设}）"
head=$(git -C "$WT" rev-parse HEAD 2>/dev/null)
[ "$head" = "$EXPECT_COMMIT" ] || die "工作树的 commit 是 ${head:-?}，任务书要 $EXPECT_COMMIT"
[ -e "$MAIN_PAUSE" ] || die "主驱动的暂停文件不在（$MAIN_PAUSE）：主驱动没有暂停，不能同时用卡"
[ -f "$MAIN_DRV/done.txt" ] || die "主驱动的记录不在（$MAIN_DRV）"
if [ "$TINY" != 1 ]; then
  case "$(cat "$LOGS/train_0051-$NODE.status" 2>/dev/null)" in *paused*) ;; *) die "主驱动的心跳不是 paused：$(cat "$LOGS/train_0051-$NODE.status" 2>/dev/null)" ;; esac
fi

tag_dir() {  # eval_<tag> 的 <tag> → 模型目录（<MAIN_OUT>/trained/<起点>/<模型>/n<数据量>[_abcd]/seed<k>；同 queue.sh 的 e5() 的标记）
  local t=$1
  [[ $t =~ ^(.+)_n([0-9]+)_s([0-9]+)(_abcd)?$ ]] || return 1
  echo "$MAIN_OUT/trained/${BASH_REMATCH[1]}/$M/n${BASH_REMATCH[2]}${BASH_REMATCH[4]}/seed${BASH_REMATCH[3]}"
}
criteria_of() {  # $1 = merge_check.json → criteria_version（没有文件或读不出时为空）
  [ -f "$1" ] && "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("criteria_version") or "")' "$1" 2>/dev/null
}
main_ckpt() { awk -F'\t' -v n="$1" '$1 == n {c = $2} END {print c}' "$MAIN_OUT/ckpt_ledger.tsv"; }

PLAN=$OUT/plan.tsv
if [ ! -f "$PLAN" ]; then  # 名单只定一次（脚本重启时照原名单接着跑，跑过的由 run_cmd 跳过）
  printf 'kind\ttag\tcheckpoint\tmodel_dir\n' > "$PLAN.tmp"
  while IFS= read -r name; do
    tag=${name#eval_}
    grep -qxF "$name" "$MAIN_DRV/done.txt" && continue
    d=$(tag_dir "$tag") || die "认不出标记 $tag"
    ck=$(main_ckpt "$tag"); [ -n "$ck" ] || die "$tag：主驱动的 ckpt_ledger.tsv 里没有检查点"
    printf 'reeval\t%s\t%s\t%s\n' "$tag" "$ck" "$d" >> "$PLAN.tmp"
  done < <(awk '{print $2}' "$MAIN_DRV/failed.txt" 2>/dev/null | grep '^eval_' | awk '!seen[$0]++')
  while IFS= read -r name; do
    tag=${name#eval_}
    d=$(tag_dir "$tag") || die "认不出标记 $tag"
    [ "$(criteria_of "$d/merge_check.json")" = e5-v2 ] && continue
    ck=$(main_ckpt "$tag"); [ -n "$ck" ] || die "$tag：主驱动的 ckpt_ledger.tsv 里没有检查点"
    printf 'check\t%s\t%s\t%s\n' "$tag" "$ck" "$d" >> "$PLAN.tmp"
  done < <(grep '^eval_' "$MAIN_DRV/done.txt" | awk '!seen[$0]++')
  mv "$PLAN.tmp" "$PLAN"
fi
note "补评开始：$NODE，commit $head；名单 $PLAN：补评 $(grep -c '^reeval' "$PLAN") 个、只核对 $(grep -c '^check' "$PLAN") 个"

while IFS=$'\t' read -r kind tag ck d <&3; do  # 名单从 3 号描述符读，命令的标准输入不碰它
  [ "$kind" = kind ] && continue
  [ -d "$ck" ] || { note "$tag：检查点不在（$ck），跳过"; echo "$(now) ${kind}_$tag missing-ckpt $ck" >> "$DRV/failed.txt"; continue; }
  task=(--task all); case "$tag" in *_abcd) task=(--task e) ;; esac  # 同 queue.sh 的 e5()：a–d 迁移的 test 只评 e
  if [ "$kind" = reeval ]; then
    EST_H=0.6
    if run_cmd "eval_$tag" "$PY" scripts/eval_e5_trained.py --checkpoint "$ck" --model $M --out "$MAIN_OUT" "${task[@]}" \
        ${EXTRA[@]+"${EXTRA[@]}"} ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}; then
      grep -qxF "eval_$tag" "$MAIN_DRV/done.txt" || echo "eval_$tag" >> "$MAIN_DRV/done.txt"
    fi
  else
    EST_H=0.2
    run_cmd "mcheck_$tag" "$PY" scripts/eval_e5_trained.py --checkpoint "$ck" --model $M --out "$MAIN_OUT" --check-only \
      ${EXTRA[@]+"${EXTRA[@]}"}
  fi
done 3< "$PLAN"

# 汇总（只读 merge_check.json：判定与指标层面的差；不读任何 test 分数）
"$PY" - "$PLAN" "$OUT/summary.tsv" <<'PYEOF'
import json, sys
from pathlib import Path
plan, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = ["\t".join(["kind", "tag", "criteria_version", "pass_e5v2", "pass_v1_report_only", "metric_status", "metric_error",
                   "d_iou_grid_all", "d_iou_grid_binary", "d_relief_error", "flag_iou", "flag_relief", "n_rows", "n_identical",
                   "previous"])]
for line in plan.read_text(encoding="utf-8").splitlines()[1:]:
    kind, tag, _ck, d = line.split("\t")
    p = Path(d) / "merge_check.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        rows.append("\t".join([kind, tag, f"unreadable: {e!r}"] + [""] * 12))
        continue
    ml = doc.get("metric_level") or {}
    fl = ml.get("flags") or {}
    def g(*ks):
        x = ml
        for k in ks:
            x = x.get(k) if isinstance(x, dict) else None
        return "" if x is None else str(x)
    err = ml.get("error")
    rows.append("\t".join([kind, tag, str(doc.get("criteria_version")), str(doc.get("pass")),
                           str((doc.get("v1_report_only") or {}).get("pass")), str(ml.get("status")),
                           "" if not err else str(err.get("repr") if isinstance(err, dict) else err).replace("\t", " ")[:200],
                           g("overall", "all", "iou_grid", "diff"), g("by_format", "binary", "all", "iou_grid", "diff"),
                           g("overall", "all", "relief_error", "diff"), str(fl.get("iou")), str(fl.get("relief")),
                           g("overall", "all", "n"), g("overall", "all", "n_identical"),
                           str((doc.get("previous") or {}).get("renamed_to") or "")]))
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(out.read_text(encoding="utf-8"))
PYEOF

rm -f "$MAIN_PAUSE" && note "删掉主驱动的暂停文件 $MAIN_PAUSE：主驱动接着跑（新判据 e5-v2）"
finish
