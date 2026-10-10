for u in https://pypi.org/simple/pip/ https://pypi.tuna.tsinghua.edu.cn/simple/pip/ https://download.pytorch.org/whl/cu130/torch/ https://hf-mirror.com/api/models/Qwen/Qwen3.5-0.8B https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B https://github.com https://www.google.com https://drive.google.com https://web.archive.org; do
  r=$(curl -s -o /dev/null -m 15 -w "%{http_code} %{time_total}s" "$u" 2>&1); echo "$(hostname -s) $u -> $r"
done
echo "$(hostname -s) ssh github:"; timeout 15 ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -T git@github.com 2>&1 | tail -1
echo "$(hostname -s) torch cu130 wheels (pytorch.org):"; curl -s -m 30 https://download.pytorch.org/whl/cu130/torch/ | grep -oE 'torch-2\.1[0-9]\.[0-9](%2B|\+)cu130-cp312-cp312-manylinux[^"#<]*x86_64\.whl' | sort -u | tail -6
echo "$(hostname -s) hf-mirror speed (0.8B safetensors HEAD + 200MB range):"
curl -sI -m 20 -L https://hf-mirror.com/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors-00001-of-00001.safetensors 2>&1 | egrep -i "^HTTP|content-length|x-repo-commit|etag" | head -6
curl -s -m 60 -L -r 0-209715199 -o /dev/null -w "range200MB http=%{http_code} size=%{size_download} speed=%{speed_download}B/s t=%{time_total}s\n" https://hf-mirror.com/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors-00001-of-00001.safetensors
