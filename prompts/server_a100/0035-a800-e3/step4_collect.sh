#!/usr/bin/env bash
# RUN-0035 第 4 步：打包结果与最终检查点（final/，不含续跑状态），传回 Google Drive。随时可以运行，回传已完成的部分。
set -euo pipefail
WORK=${1:?用法: bash step4_collect.sh <工作目录>}
TS=$(date -u +%Y%m%dT%H%MZ)
tar -czf "$WORK/a800_results_0035_$TS.tar.gz" -C "$WORK" --exclude='*.pt' results_0035
# 检查点：只要 final/（LoRA、token 行、读出），给 B200 评测（D55、D58）
( cd "$WORK" && find e3_ckpt_0035 -type d -name final -print0 | tar -czf "$WORK/a800_ckpt_0035_$TS.tar.gz" --null -T - )
ls -la "$WORK"/a800_*_0035_"$TS".tar.gz
sha256sum "$WORK"/a800_*_0035_"$TS".tar.gz 2>/dev/null || shasum -a 256 "$WORK"/a800_*_0035_"$TS".tar.gz
echo "把这两个文件和 notes.txt 上传到 Google Drive 的 vision-internalization-a100/ 文件夹。"
