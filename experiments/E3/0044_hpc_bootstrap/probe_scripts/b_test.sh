set -u
VI=/home/mail/chenyuma/visint; O=$VI/logs/0044_methodB
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; }
SQ before
echo "--- on hpcgpu06 via ssh -n: which srun / squeue"
ssh -n -o BatchMode=yes hpcgpu06 'hostname -s; which srun squeue sinfo 2>&1; echo PATH=$PATH | cut -c1-200'
echo "--- launch from node"
ssh -n -o BatchMode=yes hpcgpu06 "cd $VI && setsid nohup srun --jobid=157296 --overlap -n1 bash -c 'echo host=\$(hostname -s) CVD=\${CUDA_VISIBLE_DEVICES:-unset} nproc=\$(nproc) step=\$SLURM_JOB_ID.\$SLURM_STEP_ID; cat /proc/self/cgroup | egrep \"devices|cpuset\"; nvidia-smi -L; sleep 60; echo end \$(date -u +%FT%TZ)' > $O.log 2>&1 < /dev/null & echo launched_pid=\$!"
sleep 10; squeue -s -j 157296 -o "%i %j %M %N"; cat $O.log
echo "--- srun client process on node:"; ssh -n hpcgpu06 "pgrep -u chenyuma -af '[s]run --jobid=157296 --overlap -n1 bash -c' | cut -c1-120"
