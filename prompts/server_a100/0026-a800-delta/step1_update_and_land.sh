#!/usr/bin/env bash
# RUN-0026 第 1 步（需要联网）：在 RUN-0022 的工作目录上更新代码快照，然后落地数据。不重装环境、不重下权重。
# 用法：bash step1_update_and_land.sh <RUN-0022 的工作目录，例如 /root/autodl-tmp/visint-0022>
# - refer_zips：把 Drive 上的 refcoco.zip、refcoco+.zip、refcocog.zip 放进 <工作目录>/refer_zips/（web.archive.org 不通时必需）
# - 国内网络：脚本默认 HF_ENDPOINT=https://hf-mirror.com；能直连 huggingface.co 的话可以 export HF_ENDPOINT= 清空
set -uo pipefail
WORK=${1:?用法: bash step1_update_and_land.sh <工作目录>}
PKG=$(cd "$(dirname "$0")" && pwd)
OUT="$WORK/results_0026"; mkdir -p "$OUT/land_logs" "$WORK/tmp" "$WORK/data"
step() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1 rc=$2" | tee -a "$OUT/steps.log"; }
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
export HF_HOME=${HF_HOME:-$WORK/hf} TMPDIR=$WORK/tmp PIP_NO_CACHE_DIR=1
export HF_ENDPOINT=${HF_ENDPOINT-https://hf-mirror.com}

# 1. 换代码快照（旧的改名保留，不删）
if [ -d "$WORK/vision-internalization" ]; then mv "$WORK/vision-internalization" "$WORK/vision-internalization.before0026.$(date +%s)"; fi
tar -xzf "$PKG"/vision-internalization-*.tar.gz -C "$WORK" && cp "$PKG/VERSION.txt" "$WORK/vision-internalization/PACKAGE_VERSION.txt"
step update_code $?
cd "$WORK/vision-internalization"
pip install --dry-run -r requirements.txt 2>&1 | grep -i 'would install' > "$OUT/pip_dryrun.txt"; echo "(空 = 没有要新装的包)" >> "$OUT/pip_dryrun.txt"
pip freeze > "$OUT/pip_freeze.txt"

# 2. refer zip 的完整性（有就核对）
REFER=""
if [ -d "$WORK/refer_zips" ]; then
  python scripts/data/refer_zip_sha256.py "$WORK/refer_zips" > "$OUT/refer_zip_check.txt" 2>&1; rc=$?; step refer_zip_check $rc
  [ $rc -eq 0 ] && REFER="--refer-zip-dir $WORK/refer_zips"
else
  step refer_zip_missing 0
fi

# 3. 落地。每个源独立，失败只记日志；已落好的会跳过
python scripts/data/land_e1.py refcoco --root "$WORK/data" --tmpdir "$TMPDIR" --budget-gb 5 $REFER \
  ${HF_ENDPOINT:+--mirror-hf-base "$HF_ENDPOINT"} > "$OUT/land_logs/land_e1_refcoco.log" 2>&1; step land_e1_refcoco $?
python scripts/data/land_e1.py nyuv2 --root "$WORK/data" --tmpdir "$TMPDIR" --budget-gb 6 \
  > "$OUT/land_logs/land_e1_nyuv2.log" 2>&1; step land_e1_nyuv2 $?
for s in ade20k nyu_normals nyu_depth_train refcoco_train; do
  extra=""; [ "$s" = refcoco_train ] && extra="$REFER"
  python scripts/data/land_train.py "$s" --root "$WORK/data" --tmpdir "$TMPDIR" $extra \
    > "$OUT/land_logs/land_$s.log" 2>&1; step "land_$s" $?
done
du -sh "$WORK/data"/* > "$OUT/data_sizes.txt" 2>&1; df -h "$WORK" >> "$OUT/data_sizes.txt"
for m in "$WORK"/data/*/MANIFEST.json; do cp "$m" "$OUT/land_logs/MANIFEST_$(basename "$(dirname "$m")").json"; done
echo "[step1] 完成。下一步：bash step2_checks.sh $WORK"
