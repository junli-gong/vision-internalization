#!/usr/bin/env bash
# RUN-0052 补充 1（D98）：4B 的 e3b_s2 另起一步——E3b 队列在 e3b_s2 之前因用量上限停下（Lustre 慢启动多耗约 6 小时），run.sh 已做完
# 种子 0、1 的 eval b。这里依次：queue.sh（跳过已完成的 e3b_s0、e3b_s1，只训 e3b_s2；QUEUE_CAP_H 由任务书放宽）→ eval.sh b 只评种子 2。
# 代码换成含 DEV-0047 的 commit（训练加载器按目录列清单查文件，训练池逐条相同，续跑身份不变）。
# 铁律：任何情况下都不能取消或停止 Slurm 任务 157296、157297。
# 用法（hpcgpu06 上经 srun 启动，见 0052-addendum-1.md）：
#   srun --jobid=157296 --overlap -n1 env PATH=… QUEUE_CAP_H=… EST_E3B_H=… EV_CAP_B_H=… bash $VI/wt/wt0052/prompts/server_hpc/0052-hpc-06-4b/run_s2.sh < /dev/null
D=$(cd "$(dirname "$0")" && pwd)
st() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${LOGS:-/home/mail/chenyuma/visint/logs}/run_0052.log"; }
st "s2 start (addendum-1)"
bash "$D/queue.sh"; st "queue (s2) rc=$?"
EV_CAP_H=${EV_CAP_B_H:-} EVAL_SEEDS=2 bash "$D/eval.sh" b; st "eval b (s2) rc=$?"
st "s2 done"
