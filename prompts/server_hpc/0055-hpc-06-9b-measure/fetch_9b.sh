#!/usr/bin/env bash
# RUN-0055 第一部分：Qwen3.5-9B 的权重经 hf-mirror 落到 HPC 的 HF 缓存（$VI/hf/hub），revision 钉 weights_manifest.json 的值，
# 下载后逐文件核对字节数与 LFS sha256（同 RUN-0044 §4.3，用 RUN-0009 的 p12_verify_weights.py）。
# 只用 CPU 与网络，不用 GPU：可以在 06 训练 E5 的同时经 srun --overlap 跑。不并发下载（同一时刻只跑一份本脚本）。
# RUN-0044 的教训：hf-mirror 会把请求重定向到 Xet 的 CAS（401），要设 HF_HUB_DISABLE_XET=1；长连接可能降到约 100 KB/s，见任务书的处理办法。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法（见 0055-hpc-06-9b-measure.md §1）：srun --jobid=157296 --overlap -n1 env PATH=… bash $VI/wt/wt0055/prompts/server_hpc/0055-hpc-06-9b-measure/fetch_9b.sh < /dev/null
set -uo pipefail
VI=/home/mail/chenyuma/visint
WT=$VI/wt/wt0055
LOGS=$VI/logs
PY=$VI/env/bin/python
ST=$LOGS/fetch_0055.status
export HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub TMPDIR=$VI/tmp PIP_NO_CACHE_DIR=1
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE
st() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "$ST.tmp" && mv -f "$ST.tmp" "$ST"; }
cd "$WT" || { st "blocked: 工作树不在 $WT"; exit 1; }
V=experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
[ -f "$V" ] || { st "blocked: 核对脚本不在 $V"; exit 1; }
for i in 1 2 3; do
  st "running download qwen3.5-9b (attempt $i)"
  if "$PY" "$V" qwen3.5-9b --download; then st "done fetch_0055 (attempt $i)"; exit 0; fi
  sleep 60
done
st "failed fetch_0055 (3 attempts)"
exit 1
