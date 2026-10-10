U=https://hf-mirror.com/Qwen/Qwen3.5-2B/resolve/15852e8c16360a2fea060d615a32b45270f8a8fc/model.safetensors-00001-of-00001.safetensors
curl -sI -m 20 -L "$U" | egrep -i "^HTTP|content-length|^location" | head -6
for i in 1 2; do curl -s -L -m 30 -r $((i*100000000))-$((i*100000000+104857599)) -o /dev/null -w "$(hostname -s) range$i http=%{http_code} size=%{size_download} speed=%{speed_download}B/s\n" "$U"; done
