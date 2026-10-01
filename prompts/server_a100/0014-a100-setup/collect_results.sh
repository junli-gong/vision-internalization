#!/usr/bin/env bash
# RUN-0014 第 4 步：把结果打成一个压缩包，放回 Google Drive 的同一个文件夹。
# 用法：bash collect_results.sh <工作目录> [结果目录，默认 <工作目录>/results_0014]
set -euo pipefail
WORK=${1:?用法: bash collect_results.sh <工作目录> [结果目录]}
OUT=${2:-$WORK/results_0014}
cp "$WORK/pip_freeze.txt" "$OUT/" 2>/dev/null || true
NAME="a100_results_0014_$(date -u +%Y%m%dT%H%MZ).tar.gz"
tar -czf "$WORK/$NAME" -C "$(dirname "$OUT")" "$(basename "$OUT")"
ls -la "$WORK/$NAME"
echo "把 $WORK/$NAME 上传到 Google Drive 的 vision-internalization-a100/ 文件夹。"
