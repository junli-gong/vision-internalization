# RUN-0097 报告：hpcgpu06，外部对照 LISA-7B 在 W2 test 的 1,500 句子集上（D154、D156；预注册 U-2）

任务书 `prompts/server_hpc/0097-hpc-06-lisa.md`。LISA 只作对照单独运行，输出只在 `experiments/W2/0097_lisa_hpcgpu06`（硬约束第 1 条）。只贴数字，不解读。

## 第 1 步：工作树（hpcmgt，10-30 03:24:47Z 起，`$VI/tmp/setup0097.sh`）

- 前 squeue：157296 RUNNING 75-14:50:33 hpcgpu06；157297 RUNNING 77-12:22:42 hpcgpu07。
- fetch 第 1 次 rc=0；wt0097（稀疏）HEAD `e6012babc5db824c1f2513cd13cc8ebedf6f41ea`，status 空；`third_party/lisa/` 51 个文件，`tests/_lisa_fakes.py` 在；11 个 sha256 与任务书逐字相同（`lisa.sh` `3f4d1ca1…`）。
- Lustre 余量（只记录，D95）：开始时 10,238 GB，下完权重后 10,196 GB（`/home`）。

## 第 2 步：环境与权重（hpcmgt）

- `$VI/env/bin/python -m venv --system-site-packages $VI/lisa/venv`：rc=0。
- `pip install --no-cache-dir --only-binary :all: --no-deps -r requirements_lisa.txt --dry-run`：rc=0；`Would install` 恰好是 requirements 的 24 个包（逐名自动比对）。pip 走 hpcmgt 上 pip 配置的清华镜像；提示所钉的 `opencv-python-headless 4.8.0.74` 是 yanked 版本（"Reason for being yanked: <none given>"），照钉住的版本装（管理员确认）。正式安装 rc=0。
- `pip freeze --path $VI/lisa/venv/lib/python3.12/site-packages`：

```
accelerate==0.21.0
certifi==2024.8.30
charset-normalizer==3.4.0
contourpy==1.3.0
cycler==0.12.1
einops==0.4.1
fonttools==4.54.1
huggingface-hub==0.16.4
idna==3.10
kiwisolver==1.4.7
matplotlib==3.9.2
numpy==1.26.4
opencv-python-headless==4.8.0.74
psutil==6.1.0
pyparsing==3.1.4
python-dateutil==2.9.0.post0
requests==2.31.0
safetensors==0.4.5
scipy==1.11.2
sentencepiece==0.2.2
six==1.16.0
tqdm==4.64.1
transformers==4.31.0
urllib3==2.2.3
```

- 权重：直连 HF 03:29:49Z 开始，03:32:21Z rc=1（`httpx.ConnectTimeout: [Errno 110] Connection timed out`）；照任务书改 `HF_ENDPOINT=https://hf-mirror.com`（另设 `HF_HUB_DISABLE_XET=1`），03:50:59Z rc=0。
- `weights --verify`：rc=0，ok true。lisa `xinlai/LISA-7B-v1@43c754ee` 9 个文件 16,121,904,615 字节，clip `openai/clip-vit-large-patch14@32bd6428` 3 个文件 1,710,545,415 字节，都 `sha256_verified: true`；合计 17.832 GB。
- import 冒烟（venv 的 python，登录节点）：`run_lisa.py --help` rc=0；`hide_tokenizers` → `{'hidden': True, 'found_in_environment': True, 'found_version': '0.23.2'}`；numpy 1.26.4、cv2 4.8.0、transformers 4.31.0、torch 2.13.0+cu130、scipy 1.11.2、accelerate 0.21.0、hf_hub 0.16.4；官方 `model.LISA`、`model.llava.conversation`、`llava_llama.LlavaConfig`、`segment_anything … ResizeLongestSide`、`utils.dataset`、`utils.utils` 都 import 成功并通过 `check_module_origin`；pycocotools 在 numpy 1.26.4 下 import 成功（不需要 2.0.8 备选）。rc=0。
- 后 squeue（03:52:07Z）：157296、157297 RUNNING。

## 第 3 步：06

- 03:53:59Z 按任务书命令启动（`srun --jobid=157296 --overlap -n1 -c 8 env PATH=… TINY=0 EV_CAP_H=2 bash …/lisa.sh`），`-c 8` 被接受。前后 squeue：157296、157297 RUNNING。
- 03:54:25Z 第一步 `lisa_export`；推送循环 TAG 0097（PID 189759）。

### steps.log 原文

```
2026-10-30T03:54:25Z start commit e6012babc5db824c1f2513cd13cc8ebedf6f41ea node hpcgpu06 lisa 43c754eef75871fcc11c84d3930a402f7b0a754f clip 32bd64288804d66eefd0ccbe215aa642df71cc41
2026-10-30T03:55:46Z lisa_export rc=0 wall_s=81
2026-10-30T03:57:10Z lisa_smoke rc=0 wall_s=84
2026-10-30T04:02:37Z lisa_full rc=0 wall_s=327
2026-10-30T04:02:49Z lisa_score rc=0 wall_s=12
2026-10-30T04:02:49Z done; failed steps: 0
2026-10-30T04:02:49Z （不推送，EV_NO_GIT=1：由登录节点的同步循环推）LISA 0097 done (failed steps: 0)
```

- 心跳 `2026-10-30T04:02:49Z done 0097 (failed steps: 0)`。`logs/` 下日志无 Traceback / OOM。

### 试跑（`lisa-7b-v1/raw/logs/run_1.json`，每个划分 20 句）

- 5 个划分各写出 20 句，failures 全是 `{}`，挂起的异常 0。
- 加载 `load_s` 8.632；加载后显存峰值 15,444,875,776 字节；推理显存峰值 17,145,150,976 字节。
- 参数 7,704,747,312 个，全部 bf16 在 cuda；missing_keys []、unexpected_keys []；`seg_token_idx` 32003，分词器长度 32004。
- 每句用时（100 句的行内 `wall_s`）：中位 0.191 s，均值 0.281 s；第一句 8.99 s（热身）。试跑 `wall_s` 80.128。
- 行内核对：`seg_source` teacher_forced；`forwards` llm 1、sam_image_encoder 1、sam_mask_decoder 1。

### 计分（`lisa-7b-v1/refseg/lisa_7b/<划分>/metrics.json`；95% CI，1,000 次重抽）

| 划分 | n | cIoU [95% CI] | gIoU [95% CI] | complete | failures | n_failed | n_empty_mask | n_exif_rotated | n_image_hw_mismatch |
|---|---|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | 300 | 0.6993 [0.6512, 0.7401] | 0.7070 [0.6685, 0.7406] | True | {"ok": 300} | 0 | 2 | 0 | 0 |
| test_refcoco_unc_testB | 300 | 0.6549 [0.6096, 0.6999] | 0.6497 [0.6138, 0.6861] | True | {"ok": 300} | 0 | 1 | 0 | 0 |
| test_refcoco+_unc_testA | 300 | 0.5932 [0.5439, 0.6452] | 0.6218 [0.5845, 0.6627] | True | {"ok": 300} | 0 | 2 | 0 | 0 |
| test_refcoco+_unc_testB | 300 | 0.4826 [0.4275, 0.5359] | 0.4515 [0.4071, 0.4906] | True | {"ok": 300} | 0 | 11 | 0 | 0 |
| test_refcocog_umd_test | 300 | 0.6599 [0.6108, 0.7051] | 0.6364 [0.5959, 0.6760] | True | {"ok": 300} | 0 | 0 | 0 | 0 |

- 五个划分都是 `stand_in` False、`protocol` refer_seg_validate。
- `logs/lisa_score.log` 末尾原文：

```
LISA test_refcoco_unc_testA n=300/300 complete=True cIoU=0.6992685964416935; gIoU=0.7070335228812819 failures={'ok': 300}
LISA test_refcoco_unc_testB n=300/300 complete=True cIoU=0.6549269950082697; gIoU=0.6497425545011525 failures={'ok': 300}
LISA test_refcoco+_unc_testA n=300/300 complete=True cIoU=0.5931819662390548; gIoU=0.621833985801937 failures={'ok': 300}
LISA test_refcoco+_unc_testB n=300/300 complete=True cIoU=0.4825731461889618; gIoU=0.45148643928727317 failures={'ok': 300}
LISA test_refcocog_umd_test n=300/300 complete=True cIoU=0.6598670872488516; gIoU=0.6363664715544093 failures={'ok': 300}
```

## 收尾

- 推送循环 0097：在睡眠中按 PID 停（189759），ONCE 推送 04:04:27Z ok。
- 后 squeue（04:04:27Z）：157296 RUNNING 75-15:30:13 hpcgpu06；157297 RUNNING 77-13:02:22 hpcgpu07。
- 06 现在空闲。`$VI/lisa/venv` 与 `$VI/hf/hub` 下的两份权重留在 HPC。
