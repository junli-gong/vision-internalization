#!/usr/bin/env bash
# RUN-0029 第 1 步：换代码快照（数据、venv、权重都不动），在 GPU 上跑上采样的单元测试（含 CUDA 用例）。约 5 分钟。
PKG=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$PKG/common.sh" "$@"
if [ "$TINY" != 1 ]; then
  if [ -d "$CODE" ]; then mv "$CODE" "$CODE.before0029.$(date +%s)"; fi
  tar -xzf "$PKG"/vision-internalization-*.tar.gz -C "$WORK" && cp "$PKG/VERSION.txt" "$CODE/PACKAGE_VERSION.txt"
  step update_code $?
  cd "$CODE" || exit 1
  pip install --dry-run -r requirements.txt 2>&1 | grep -i 'would install' > "$OUT/pip_dryrun.txt"; echo "(空 = 没有要新装的包)" >> "$OUT/pip_dryrun.txt"
  { date -u; hostname; cat PACKAGE_VERSION.txt; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h "$WORK"; } > "$OUT/facts.txt" 2>&1
fi
cd "$CODE" || exit 1
python -m pytest tests/test_e2_bilinear.py -q -rs -p no:cacheprovider > "$OUT/test_e2_bilinear.log" 2>&1
step test_e2_bilinear $?
tail -3 "$OUT/test_e2_bilinear.log"
echo "[step1] 完成。下一步：bash step2_estimate.sh $WORK"
