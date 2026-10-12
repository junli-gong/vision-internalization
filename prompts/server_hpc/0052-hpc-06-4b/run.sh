#!/usr/bin/env bash
# RUN-0052（ITSC HPC hpcgpu06）：4B 的评测与 E3b × 3（D91、D93；E3 预注册 §12、§13）。依次跑三段，前一段失败不挡后一段：
#   1. eval.sh a：RUN-0045 的检查点（C3、E3b+C3 × 3 的 T 文本全套，E3b+C3 × 3 的结构强制网格）与 4B 基座的 T 文本全套；
#   2. queue.sh：4B 的 E3b × 3 训练；
#   3. eval.sh b：E3b × 3 的结构强制网格。
# 每段各有自己的上限（任务书 §3）：EV_CAP_A_H、QUEUE_CAP_H（以及 EST_E3B_H）、EV_CAP_B_H。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法（hpcgpu06 上经 srun 启动，见任务书 §4）：
#   srun --jobid=157296 --overlap -n1 env PATH=… EV_CAP_A_H=… QUEUE_CAP_H=… EST_E3B_H=… EV_CAP_B_H=… bash $VI/wt/wt0052/prompts/server_hpc/0052-hpc-06-4b/run.sh < /dev/null
D=$(cd "$(dirname "$0")" && pwd)
st() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${LOGS:-/home/mail/chenyuma/visint/logs}/run_0052.log"; }
st "start"
EV_CAP_H=${EV_CAP_A_H:-} bash "$D/eval.sh" a; st "eval a rc=$?"
bash "$D/queue.sh"; st "queue rc=$?"
EV_CAP_H=${EV_CAP_B_H:-} bash "$D/eval.sh" b; st "eval b rc=$?"
st "done"
