#!/usr/bin/env bash
# RUN-0054（ITSC HPC hpcgpu07）：0.8B 秩 128 的 C3 与 E3b+C3 × 3（D99-1，E3 预注册 §13a 的修订）。依次两段，前一段失败不挡后一段：
#   1. queue.sh：训练（C3 r128、E3b+C3 r128，种子 0 → 1 → 2）；
#   2. eval.sh：两臂的 T 文本全套与 E3b+C3 的结构强制网格。
# 上限：QUEUE_CAP_H（以及 EST_C3_H、EST_MIX_H）、EV_CAP_H。铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
# 用法：srun --jobid=157297 --overlap -n1 env PATH=… QUEUE_CAP_H=… EST_C3_H=… EST_MIX_H=… EV_CAP_H=… bash $VI/wt/wt0054/prompts/server_hpc/0054-hpc-07-r128/run.sh < /dev/null
D=$(cd "$(dirname "$0")" && pwd)
st() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${LOGS:-/home/mail/chenyuma/visint/logs}/run_0054.log"; }
st "start"
bash "$D/queue.sh"; st "queue rc=$?"
bash "$D/eval.sh"; st "eval rc=$?"
st "done"
