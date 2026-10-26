[RUN-TASK 0078] hpcgpu06：外部基线 LLMSeg，官方代码配本地 Qwen3.5-9B 零样本，跑 W2 的 test 子集 5 × 300 句（W2 预注册 §13、§13a、§13b；D125-4、D135）。约 7 GPU 小时，上限 12 小时。现在就发，驱动自动排在 W2-c 补跑（RUN-0072 补充 1）之后

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰其他在跑的 RUN。
- 只用 06 自己的暂停文件。
- **不调用任何闭源 API**：官方代码里的 Gemini、Anthropic、OpenAI、OpenRouter 调用已在运行时替换为抛错，不要设任何 API key。

## 为什么

- 负责人要求的外部基线（D125-4）：LLMSeg（arXiv 2610.29409）用官方代码，配 Qwen3.5-9B 零样本，跑与 W2-c 相同的 test 子集。
- **如果它在 9B 上不低于 W2-c，管理员要立即报告负责人。** 触发规则在预注册 §13、§13a，由管理员读，你只贴数字。

## 文件（commit 见发出时的启动语；工作树 wt0078 建在 origin/main）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0078-hpc-06-llmseg/llmseg.sh` | `8a6a4d34410d703bb322b2abefcdf8d2e3a11e2abf1c40375791c8581b4439c1` |
| `scripts/run_llmseg.py` | `510b66416cf6cb0a1411582f848d607f0f4b6bd1223f414446f027b3703fb887` |
| `scripts/compare_llmseg.py` | `f697bb75351b2c666f30a274af3e72b97fdf90c86d675950bb0da8d5c2f7157a` |
| `src/llmseg/__init__.py` | `b3d443d0f7f11bfee470270bc9ce2c61d1cb438274a6db956996dc30a584163a` |
| `src/llmseg/vendor.py` | `adcfc852cd9c24364cbdef9b95ff7c69cf53bbc13f851d089f44493e420f4b4b` |
| `src/llmseg/provider.py` | `aeac30e3fe7ee3051bebd8dc316076b6004c331161cbbd80c9a00a6fee50b178` |
| `src/llmseg/runner.py` | `c0d47365b56b1bbd255ceb96df131f8fbbd63408436e145ecbe90e826d3298e6` |
| `third_party/llmseg/PROVENANCE.md` | `6be8e7382736e8f0a31ad7fdd1fe0ac7105e07d647169c7c237d301ed266240d` |
| `configs/w2_llmseg_subsets.json` | `dfd53ce8e9a89d85ba828de6c0b3e331f83bfcf7d476cfc7ed1751e64c7497ae` |
| `configs/w2_splits.json` | `9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1` |
| `src/adapters/qwen35.py` | `64d2c6e0e4deb7cf41a44dbf1173742944a4c2d691ffd8d213b6ed3d61bdca99` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

- `third_party/llmseg/` 的 8 个官方文件由 `src/llmseg/vendor.py` 在运行时逐个对照 `PROVENANCE.md` 核对 sha256，对不上就拒跑。
- 工作树是稀疏检出，要包含 `third_party/`。

## 内容（`llmseg.sh`）

1. **排队**：
   - 先等 `$LOGS/eval_0072r.status` 以 done 开头，即 W2-c 补跑完成。若它是 blocked，就不开跑。等待上限 48 小时。
   - 然后 `claim_gpu`：GPU 上没有进程才开始。
2. **llmseg**：`run_llmseg.py --split all`。
   - 5 个 test 划分各 300 句，官方主配置 T = 8，关思考、贪心。
   - 合批：slots 150、batch 32、max_batch_tokens 48000、chunk 150，可续跑。
   - 输出 `experiments/W2/0078_llmseg_hpcgpu06/qwen3.5-9b/refseg/llmseg/<划分>/`。每句的工作目录 `work/` 不入库（.gitignore）。
3. **compare**：`compare_llmseg.py`，与 RUN-0072 的 W2-c（`0072_w2_test_hpcgpu06/qwen3.5-9b/refseg/w2c`）在同样的 1,500 句上配对，seg 与 rec 两组。

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0078（稀疏，含 `third_party/`），建在 origin/main。fetch 失败就重试。核对上表 12 个 sha256。
2. **启动**（方案 B）：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=12 \
     bash /home/mail/chenyuma/visint/wt/wt0078/prompts/server_hpc/0078-hpc-06-llmseg/llmseg.sh \
     > /home/mail/chenyuma/visint/logs/eval_0078.driver.log 2>&1 < /dev/null &'
   ```
   - 心跳在 `$VI/logs/eval_0078.status`，等待期间是 "waiting for eval_0072r.status to be done"。
   - 驱动开始跑之后，起一个 v3 推送循环，TAG 0078，路径 `experiments/W2/0078_llmseg_hpcgpu06`，排除 `work/`。
3. **第一组 150 句跑完报一次**：
   - `logs/gpu_summary.json` 的外推与显存峰值；
   - 每图实际调用次数的分布；
   - 各阶段的截断率与格式失败率；
   - 坐标 > 1000 的计数；
   - 墙钟。
   - **外推的总墙钟超过 12 小时，或显存不够（OOM）**，就停掉本任务的 python（不碰 Slurm 任务），报告，等我回复。显存不够时我会让你用 `LLMSEG_ARGS="--batch-size 16"` 加续跑重启。
4. 每个划分跑完报一次：seg 与 rec 的 cIoU、gIoU 与 CI，失败率。
5. compare 跑完，把 `compare_w2c` 的 markdown 原文照贴，包括 `flag_llmseg_not_lower` 一项。不作解读。
6. 全部跑完：写报告，再跑一次 `squeue`。

## 报告

推送 `reports/0078-hpc-06-llmseg.md`，内容：
- `steps.log` 原文与墙钟；
- 第 3 步的吞吐与显存；
- 各划分的 seg 与 rec 读数；
- 计算记账：调用数、token、每图调用次数的分布、截断率、格式失败率、坐标统计；
- compare 的原文；
- 显存峰值、任何 traceback；
- 确认全程没有任何外部 API 调用：日志里没有 `ExternalCallForbidden`，若出现过就照实报。
