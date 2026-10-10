SQ() { ssh a100 'ssh gpu "date -u +%FT%TZ; squeue -h -j 157296,157297 -o \"%i %T %M %N\""' 2>&1 | grep -v libmamba; }
{ echo "## squeue before"; SQ
  echo "===== hpcmgt"; ssh a100 'ssh gpu bash -s' < nettest_inner.sh 2>&1 | grep -v libmamba
  for J in 157296 157297; do echo "===== job $J (nettest)"; ssh a100 "ssh gpu 'timeout 400 srun --jobid=$J --overlap -n1 bash -s'" < nettest_inner.sh 2>&1 | grep -v libmamba
    echo "===== job $J (dataset hosts)"; ssh a100 "ssh gpu 'timeout 400 srun --jobid=$J --overlap -n1 bash -s'" < net2_inner.sh 2>&1 | grep -v libmamba; done
  echo "## squeue after"; SQ; } > r_net.out 2>&1
{ echo "## squeue before"; SQ
  for J in 157296 157297; do echo "##### $J"; ssh a100 "ssh gpu 'timeout 120 srun --jobid=$J --overlap -n1 bash -s'" < ps_inner.sh 2>&1 | grep -v libmamba; ssh a100 "ssh gpu 'timeout 120 srun --jobid=$J --overlap -n1 bash -s'" < p_inner.sh 2>&1 | grep -v libmamba; done
  echo "## hpcmgt"; ssh a100 'ssh gpu "uptime; hostname"' 2>&1 | grep -v libmamba
  echo "## squeue after"; SQ; } > r_ps.out 2>&1
{ echo "b200->mac 100MB:"; /usr/bin/time -p ssh b200 'head -c 104857600 /dev/zero' 2>&1 >/dev/null | tail -3
  echo "mac->lab5 50MB:"; /usr/bin/time -p sh -c 'head -c 52428800 /dev/urandom | ssh a100 "cat > /dev/null"' 2>&1 | tail -3
  echo "lab5->hpcmgt 50MB:"; ssh a100 '/usr/bin/time -p sh -c "head -c 52428800 /dev/urandom | ssh gpu \"cat > /dev/null\""' 2>&1 | grep -v libmamba | tail -3
  echo "lab5 internet:"; ssh a100 'for u in http://images.cocodataset.org/train2014/COCO_train2014_000000581857.jpg https://hf-mirror.com/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors-00001-of-00001.safetensors; do curl -s -L -m 25 -r 0-52428799 -o /dev/null -w "lab5 $u http=%{http_code} size=%{size_download} speed=%{speed_download}B/s\n" "$u"; done'
  echo "b200 -> pi.machenyu.cn:8889 tcp:"; ssh b200 'timeout 10 bash -c "</dev/tcp/pi.machenyu.cn/8889" && echo open || echo closed'
  echo "b200 train_data:"; ssh b200 'du -s --apparent-size -B1 /dev/shm/visint/train_data/*/ ; ls -la /dev/shm/visint/train_data/'
} > r_relay.out 2>&1
echo done
