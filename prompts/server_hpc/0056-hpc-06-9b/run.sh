#!/usr/bin/env bash
# RUN-0056（ITSC HPC hpcgpu06）：9B 的方向性检查（E3 预注册 §3；N41 批准后）。依次跑四段，前一段失败不挡后一段：
#   1. queue.sh 1：c3_s0 → e3b_c3_s0；
#   2. eval.sh a：E3b+C3、C3 的 T 文本全套，E3b+C3 的结构强制网格，9B 基座的 T 文本全套；
#   3. queue.sh 2：e3b_s0；
#   4. eval.sh b：E3b 的结构强制网格。
# 先出 L（语言干扰，§13）需要的两臂，再训 E3b。每段各有自己的上限（任务书 §3）：QUEUE_CAP_1_H、EV_CAP_A_H、QUEUE_CAP_2_H（累计）、EV_CAP_B_H。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
#
# 用法（hpcgpu06 上经 srun 启动，见任务书 §4）：
#   srun --jobid=157296 --overlap -n1 env PATH=… QUEUE_CAP_1_H=… EST_C3_H=… EST_MIX_H=… EV_CAP_A_H=… QUEUE_CAP_2_H=… EST_E3B_H=… EV_CAP_B_H=… \
#     bash $VI/wt/wt0056/prompts/server_hpc/0056-hpc-06-9b/run.sh < /dev/null
D=$(cd "$(dirname "$0")" && pwd)
st() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${LOGS:-/home/mail/chenyuma/visint/logs}/run_0056.log"; }
st "start"
QUEUE_CAP_H=${QUEUE_CAP_1_H:-} bash "$D/queue.sh" 1; st "queue 1 rc=$?"
EV_CAP_H=${EV_CAP_A_H:-} bash "$D/eval.sh" a; st "eval a rc=$?"
QUEUE_CAP_H=${QUEUE_CAP_2_H:-} bash "$D/queue.sh" 2; st "queue 2 rc=$?"
EV_CAP_H=${EV_CAP_B_H:-} bash "$D/eval.sh" b; st "eval b rc=$?"
st "done"
