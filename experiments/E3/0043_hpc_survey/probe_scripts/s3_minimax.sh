set -u
B=/home/mail/chenyuma/minimax-h3-fp8
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; }
SQ before
for f in minimax-h3-a100-1gpu-06.sbatch minimax-h3-a100-1gpu-07.sbatch minimax-h3-a100-1gpu-fp16.env start_minimax_h3_a100_1gpu_fp16.sh; do echo "=================== $f"; ls -la --time-style=full-iso $B/$f; cat $B/$f; done
echo "=================== run/"; ls -la --time-style=full-iso $B/run/
echo "=================== ~/h3_guard.sh"; cat ~/h3_guard.sh
echo "=================== ~/gpu.sh"; cat ~/gpu.sh
echo "=================== schedule_a100_h3.sh"; cat $B/schedule_a100_h3.sh
echo "=================== crontab -l"; crontab -l 2>&1
echo "=================== tail gpu_park.log"; tail -5 ~/gpu_park.log; ls -la --time-style=full-iso ~/gpu_park.log
echo "=================== slurm log tails"; ls -la $B/logs/ | tail; tail -20 $B/logs/slurm-h3-fp16-a100-06-157296.log; echo ...; tail -20 $B/logs/slurm-h3-fp16-a100-07-157297.log
SQ after
