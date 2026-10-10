set -u
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; }
SQ before
echo "--- quota"; timeout 60 lfs quota -hu chenyuma /home
echo "--- groupquota"; timeout 60 lfs quota -hg "$(id -gn)" /home 2>&1 | head -5
echo "--- home listing (ls only)"; ls -la /home/mail/chenyuma/ | head -50
echo "--- minimax dir (ls only)"; ls -la /home/mail/chenyuma/minimax-h3-fp8/ | head -60
echo "--- conda"; ls /home/mail/chenyuma/soft/miniconda3/bin/ | egrep "^(conda|python|python3|mamba|pip)" ; /home/mail/chenyuma/soft/miniconda3/bin/conda --version 2>&1 | grep -v libmamba; /home/mail/chenyuma/soft/miniconda3/bin/python -V; /home/mail/chenyuma/soft/miniconda3/bin/conda env list 2>&1 | grep -v libmamba
echo "--- conda pkgs dir / channels"; /home/mail/chenyuma/soft/miniconda3/bin/conda config --show pkgs_dirs channels envs_dirs 2>&1 | grep -v libmamba; cat ~/.condarc 2>/dev/null
echo "--- pip config"; cat ~/.pip/pip.conf ~/.config/pip/pip.conf 2>/dev/null; env | egrep -i "proxy|HF_|PIP_" 
echo "--- tools"; which git tmux python3 module 2>&1; git --version; tmux -V; python3 -V
echo "--- slurm partition / preempt"; scontrol show partition gpu | head -20; scontrol show config | egrep -i "preempt|JobRequeue|KillOnBadExit|OverTimeLimit|ProctrackType|TaskPlugin|SelectTypeParameters|Prolog|Epilog" 
echo "--- my steps"; squeue -s -j 157296,157297
echo "--- other jobs on our nodes"; squeue -w hpcgpu06,hpcgpu07 -o "%i %u %j %T %M %b"
echo "--- node info"; scontrol show node hpcgpu06 | egrep "Gres|AllocTRES|CfgTRES|State|RealMemory"; scontrol show node hpcgpu07 | egrep "Gres|AllocTRES|CfgTRES|State|RealMemory"
echo "--- github ssh keys in ~/.ssh (names only)"; ls ~/.ssh/
SQ after
