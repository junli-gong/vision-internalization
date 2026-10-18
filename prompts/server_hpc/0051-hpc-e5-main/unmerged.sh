#!/usr/bin/env bash
# RUN-0051 补充 2（D99-2）：H5 判定用到的 12 个运行（E3b+C3 与 C3+预热，1K、4K，各 3 个种子）用未合并的 E5 LoRA 在 test 上再评一次
# （scripts/eval_e5_trained.py --unmerged，DEV-0048；结构强制解码，其余同正式评测）。名单现在固定，与结果无关。
# 在主驱动暂停期间（$LOGS/train.pause.$NODE 存在）、在训练与正式评测这 12 个运行的同一台节点（hpcgpu07）上经 srun 跑（D93-1）。
# 跑完删掉主驱动的暂停文件，主驱动接着跑。结果写进主驱动的结果目录（同一个 --out，叶目录 <任务>__structured__unmerged，不碰正式评测）；
# 本脚本的驱动记录放在 <MAIN_OUT>/unmerged_d99/（done.txt、failed.txt、gpu_seconds、logs/），不占主驱动的用量上限。
# 不读、不汇总任何 test 分数（D96-1 的盲：到门报告再看）；最后只写 <OUT>/summary.tsv：每个运行的叶目录是否齐、运行记录里 LoRA 审计的状态。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法（在节点上经 srun 启动，见 0051-addendum-2.md）：
#   srun --jobid=<任务> --overlap -n1 env NODE=hpcgpu07 EXPECT_COMMIT=<40 位> QUEUE_CAP_H=11 PATH=… bash $VI/wt/wt0051/prompts/server_hpc/0051-hpc-e5-main/unmerged.sh < /dev/null
# 彩排（开发机）：TINY=1 TINY_MODEL=… UNMERGED_TAGS="…" NODE=hpcgpu07 EXPECT_COMMIT=… WT=… REPO=… MAIN_OUT=… CK_TMP=… CK_FINAL=… LOGS=… PY=… bash unmerged.sh
NODE=${NODE:?NODE=hpcgpu06 或 hpcgpu07}
case "$NODE" in hpcgpu06|hpcgpu07) ;; *) echo "未知节点 $NODE" >&2; exit 2 ;; esac
EXPECT_COMMIT=${EXPECT_COMMIT:?任务书给的 commit}
RUN_ID=0051u-$NODE
M=qwen3.5-2b
# D99-2 的范围：E3b+C3 与 C3+预热 × 1K、4K × 种子 0、1、2（标记同 queue.sh 的 e5()：<起点>_n<数据量>_s<种子>）
TAGS=(${UNMERGED_TAGS:-e3b_c3_n1000_s0 c3_warmup_n1000_s0 e3b_c3_n4000_s0 c3_warmup_n4000_s0
                       e3b_c3_n1000_s1 c3_warmup_n1000_s1 e3b_c3_n4000_s1 c3_warmup_n4000_s1
                       e3b_c3_n1000_s2 c3_warmup_n1000_s2 e3b_c3_n4000_s2 c3_warmup_n4000_s2})
if [ "${TINY:-0}" = 1 ]; then
  : "${WT:?}" "${REPO:?}" "${MAIN_OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${LOGS:?}" "${PY:?}"
  CHUNK=1; EXTRA=(--tiny); EVAL_LIMIT=(--limit 2)
  M=${TINY_MODEL:-qwen3.5-0.8b}
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
OUT=$MAIN_OUT/unmerged_d99
MAIN_DRV=$MAIN_OUT/driver
MAIN_PAUSE=$LOGS/train.pause.$NODE
export TQ_PAUSE=$LOGS/train.pause.$NODE.unmerged  # 本脚本自己的暂停文件（主驱动的暂停文件在跑的期间一直存在，跑完由本脚本删掉）
export MEM_STOP_GB 2>/dev/null
# shellcheck disable=SC1091
source "$WT/scripts/server/train_queue.sh"
cd "$WT" || die "工作树不存在：$WT"
[ "$TINY" = 1 ] || [ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || die "不在 srun 的资源范围里（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设}）"
head=$(git -C "$WT" rev-parse HEAD 2>/dev/null)
[ "$head" = "$EXPECT_COMMIT" ] || die "工作树的 commit 是 ${head:-?}，任务书要 $EXPECT_COMMIT"
[ -e "$MAIN_PAUSE" ] || die "主驱动的暂停文件不在（$MAIN_PAUSE）：主驱动没有暂停，不能同时用卡"
if [ "$TINY" != 1 ]; then
  case "$(cat "$LOGS/train_0051-$NODE.status" 2>/dev/null)" in *paused*) ;; *) die "主驱动的心跳不是 paused：$(cat "$LOGS/train_0051-$NODE.status" 2>/dev/null)" ;; esac
fi
main_ckpt() { awk -F'\t' -v n="$1" '$1 == n {c = $2} END {print c}' "$MAIN_OUT/ckpt_ledger.tsv"; }
tag_dir() {  # <tag> → 模型目录（同 reeval.sh）
  [[ $1 =~ ^(.+)_n([0-9]+)_s([0-9]+)$ ]] || return 1
  echo "$MAIN_OUT/trained/${BASH_REMATCH[1]}/$M/n${BASH_REMATCH[2]}/seed${BASH_REMATCH[3]}"
}

# 开始前核对：12 个运行都已训完、正式评测都完成（主驱动 done.txt 里有 eval_<tag>），检查点在
for tag in "${TAGS[@]}"; do
  grep -qxF "eval_$tag" "$MAIN_DRV/done.txt" || die "$tag：正式评测还没完成（主驱动 done.txt 里没有 eval_$tag）"
  ck=$(main_ckpt "$tag"); [ -n "$ck" ] && [ -d "$ck" ] || die "$tag：检查点不在（${ck:-ledger 里没有}）"
done
note "未合并补评开始：$NODE，commit $head，${#TAGS[@]} 个运行：${TAGS[*]}"

for tag in "${TAGS[@]}"; do
  EST_H=0.75
  run_cmd "unmerged_$tag" "$PY" scripts/eval_e5_trained.py --unmerged --checkpoint "$(main_ckpt "$tag")" --model $M --out "$MAIN_OUT" \
    --task all ${EXTRA[@]+"${EXTRA[@]}"} ${EVAL_LIMIT[@]+"${EVAL_LIMIT[@]}"}
done

# 汇总（不读任何分数：只看叶目录在不在、运行记录里 LoRA 审计的状态）
{
  printf 'tag\tn_leaves_unmerged\taudit_status\n'
  for tag in "${TAGS[@]}"; do
    d=$(tag_dir "$tag") || { printf '%s\t?\t?\n' "$tag"; continue; }
    n=$(find "$d" -maxdepth 1 -type d -name '*__structured__unmerged' 2>/dev/null | wc -l | tr -d ' ')
    a=$("$PY" - "$d" <<'PYEOF'
import json, sys
from pathlib import Path
st = set()
for leaf in Path(sys.argv[1]).glob("*__structured__unmerged"):
    runs = sorted((leaf / "logs").glob("run_*.json"))
    if not runs:
        st.add("no_run_log"); continue
    st.add(str((json.loads(runs[-1].read_text()).get("lora_audit") or {}).get("status")))
print(",".join(sorted(st)) or "none")
PYEOF
)
    printf '%s\t%s\t%s\n' "$tag" "$n" "$a"
  done
} > "$OUT/summary.tsv"
cat "$OUT/summary.tsv"

rm -f "$MAIN_PAUSE" && note "删掉主驱动的暂停文件 $MAIN_PAUSE：主驱动接着跑"
finish
