#!/bin/bash
# RUN-0044 §4 weights: download via hf-mirror at pinned revisions, verify each file (LFS sha256 / git blob sha1) with RUN-0009's p12 verifier. One node, sequential.
export VI=/home/mail/chenyuma/visint
export HF_HOME=$VI/hf
export HF_HUB_CACHE=$VI/hf/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_CACHE=$VI/tmp/xet
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export TMPDIR=$VI/tmp
export XDG_CACHE_HOME=$VI/cache/hpcgpu06/xdg
P=$VI/probe/0044/weights
echo "== $(date -u +%FT%TZ) host=$(hostname -s)"
for m in qwen3.5-0.8b qwen3.5-2b qwen3.5-4b; do
  for a in 1 2 3; do t0=$(date +%s); $VI/env/bin/python $P/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py $m --download 2>&1 | grep -vE "it/s\]|Warn"; rc=${PIPESTATUS[0]}
    echo "=== $(date -u +%FT%TZ) $m attempt $a verify rc=$rc $(( $(date +%s)-t0 ))s"; [ $rc -eq 0 ] && break; done
done
echo "== done $(date -u +%FT%TZ)"
