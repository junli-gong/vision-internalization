#!/usr/bin/env bash
# RUN-0022 第 4 步：把结果打成一个压缩包，传回 Google Drive 的同一个文件夹。
# 用法：bash collect_results.sh <工作目录> [结果目录，默认 <工作目录>/results_0022]
set -euo pipefail
WORK=${1:?用法: bash collect_results.sh <工作目录> [结果目录]}
OUT=${2:-$WORK/results_0022}
NAME="a800_results_0022_$(date -u +%Y%m%dT%H%MZ).tar.gz"
tar -czf "$WORK/$NAME" -C "$(dirname "$OUT")" "$(basename "$OUT")"
ls -la "$WORK/$NAME"
echo "把 $WORK/$NAME 和 notes.txt 上传到 Google Drive 的 vision-internalization-a100/ 文件夹。"
