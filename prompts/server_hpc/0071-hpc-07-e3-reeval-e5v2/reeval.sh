#!/usr/bin/env bash
# RUN-0071（ITSC HPC hpcgpu07）：N37(b)——E3 结构强制网格评测按 e5-v2 判据补评被 v1 拒绝的两个种子，连同机内配对（D102-2、D113-2、D123）。
#   2B 种子 1：E3b 与 E3b+C3 两臂都在 07 上重评；4B 种子 2：E3b 与 E3b+C3 两臂都在 07 上评。每臂 refseg 与 depth，结构强制、k = 1。
#   每个种子先评 E3b；E3b 按 e5-v2 也被拒（退出码 1）就跳过那个种子的 E3b+C3（配不成对）。
#   标"按 N34 统一判据，决定于看到被拒种子的读数之前"（D102-2）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。只用本节点自己的暂停文件。
# 用法：srun --jobid=157297 --overlap -n1 env PATH=… EV_CAP_H=… bash …/reeval.sh < /dev/null
RUN_ID=0071
NODE=hpcgpu07
VI=/home/mail/chenyuma/visint
WT=$VI/wt/wt0071
REPO=$VI/repo
LOGS=$VI/logs
PY=$VI/env/bin/python
export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1
export TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1 NLTK_DATA=$VI/b200/nltk_data
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$VI/cache/$NODE/triton TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor XDG_CACHE_HOME=$VI/cache/$NODE/xdg
export EV_PAUSE=$LOGS/train.pause.$NODE EV_STATUS_GLOB="nonexistent_queue.status" EV_NO_GIT=1 EV_DF_PATH=$VI
EV_CAP_S=$(awk -v h="${EV_CAP_H:?任务书的上限（小时）}" 'BEGIN{print int(h * 3600)}')
OUT=$REPO/experiments/E3/0071_e3_reeval_e5v2_$NODE
# shellcheck disable=SC1091
source "$WT/scripts/server/eval_lib.sh"
cd "$WT" || exit 1
[ "${CUDA_VISIBLE_DEVICES:-}" = 0 ] || { progress "blocked: 不在 srun 的资源范围里"; exit 1; }
G=(--data-root "$VI/b200/data" --out "$OUT" --k-values 1 --decode structured --merge-criteria e5-v2)
NOK=0
# 检查点（manifest sha256 与台账一致，任务书核对过）
CK_2B_E3B=$VI/b200/ckpt/0036/e3b/qwen3.5-2b/seed1/ckpt/final/model
CK_2B_MIX=$VI/b200/ckpt/0040/e3b_c3/qwen3.5-2b/seed1/ckpt/final/model
CK_4B_E3B=$VI/ckpt/hpcgpu06/0052/e3b/qwen3.5-4b/seed2/ckpt/final/model
CK_4B_MIX=$VI/ckpt/hpcgpu06/0045/e3b_c3/qwen3.5-4b/seed2/ckpt/final/model
check_ck() {  # $1 = 检查点目录，$2 = 期望的 manifest sha256
  local got; got=$(sha "$1/manifest.json" 2>/dev/null | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "$(now) ABORT: $1 manifest sha256 ${got:-缺} ≠ $2" | tee -a "$OUT/steps.log"; progress "blocked: 检查点核对失败 $1"; exit 1; }
}
mkdir -p "$OUT"
check_ck "$CK_2B_E3B" 7c3d293818e377888a317753b77539dcf8c9567ae7fb01273788f5821163db57
check_ck "$CK_2B_MIX" 7c72a5007e461cca1bbb474aaaa72ba701af850587337d5ba02ee5e7ecdbb90d
check_ck "$CK_4B_E3B" dca97e758611baef5cda924b359ddae503128e78a2f7380e2ac42286a5828f12
check_ck "$CK_4B_MIX" 1b9fc10686431a95a27f87320e520d48b362b5e010ab1d748ddadef34562b11a

claim_gpu "0071-hpc-07-e3-reeval-e5v2.md" || { progress "blocked: gpu not free after ${CLAIM_MAX_S}s"; exit 1; }
echo "$(now) start commit $(git -C "$WT" rev-parse HEAD 2>/dev/null || echo unknown) node $NODE" >> "$OUT/steps.log"
pair() {  # $1 = 模型，$2 = 种子，$3 = E3b 检查点，$4 = E3b+C3 检查点
  local m=$1 s=$2 ok=1 t
  for t in refseg depth; do
    step "grid_e3b_${m}_s${s}_$t" "$PY" scripts/eval_e3b.py --checkpoint "$3" --model "$m" --variant e3b --seed "$s" "${G[@]}" --task $t || { ok=0; NOK=$((NOK + 1)); }
  done
  if [ $ok = 0 ]; then echo "$(now) skip e3b_c3 ${m} s${s}: E3b 没有读数（按 e5-v2 也被拒或出错），配不成对" >> "$OUT/steps.log"; return; fi
  for t in refseg depth; do
    step "grid_e3b_c3_${m}_s${s}_$t" "$PY" scripts/eval_e3b.py --checkpoint "$4" --model "$m" --variant e3b_c3 --seed "$s" "${G[@]}" --task $t || NOK=$((NOK + 1))
  done
}
pair qwen3.5-2b 1 "$CK_2B_E3B" "$CK_2B_MIX"
pair qwen3.5-4b 2 "$CK_4B_E3B" "$CK_4B_MIX"
echo "$(now) done; failed steps: $NOK" >> "$OUT/steps.log"
progress "done $RUN_ID (failed steps: $NOK)"
push_out "N37(b) E3 re-eval e5-v2 done (failed steps: $NOK)"
