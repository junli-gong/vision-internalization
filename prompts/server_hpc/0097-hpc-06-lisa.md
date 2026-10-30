[RUN-TASK 0097] hpcgpu06：外部对照 LISA-7B 在 W2 test 的 1,500 句子集上（D154、D156；预注册 U-2）。先在登录节点建环境、下权重，再在 06 上跑。GPU 约 0.5 小时，上限 2 小时

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 只用 06 自己的暂停文件。

## 为什么

- 负责人要求统一主表加 LISA-7B（D154）。它就是简报 §8.8 的"LISA 式对照系统"。
- **硬约束第 1 条**：LISA 只作为对照单独运行。它的输出只在 `experiments/W2/0097_lisa_hpcgpu06`，只用来计分出表，不进任何训练数据、标签或奖励。
- 句子是 `configs/w2_llmseg_subsets.json` 的 1,500 句，与 LLMSeg 相同；批准编号 D154，运行器自己核对。
- 协议照官方 refer_seg 评测（teacher forcing 给 `[SEG].`），出处见 `third_party/lisa/PROVENANCE.md`。

## 文件（工作树 wt0097 建在 origin/main；`third_party/lisa/` 整个目录要检出）

| 文件 | sha256 |
|---|---|
| `prompts/server_hpc/0097-hpc-06-lisa/lisa.sh` | `3f4d1ca1328337b3afe5b22da38bd3fcb98b293ca295c985725d2d65bf379bd5` |
| `scripts/run_lisa.py` | `60631ca9ee23d8b3d0dad1ab9beeeaa6722a8036991b48b09340118069d8cd3c` |
| `scripts/lisa_w2.py` | `4571656d7f10d72f24279d18c67d5c1f38dec9e0189fc30f73a38f538571e65d` |
| `src/lisa/common.py` | `62c48823424bbb91b67557fe8aa6341db3106bdaf7439051a8e900a66ad0f9de` |
| `src/lisa/data.py` | `287a467d43c35cd383436a972dad51271a28f77c804667f8b09efea19a031c07` |
| `src/lisa/scoring.py` | `fc583b7142d48a7cf634fa270c192c58ba91696fb4aa002803c2a34b7fa1864a` |
| `configs/lisa_weights.json` | `8076a51629d63e17b5f99e91f268ac6821eb691d62ed66f5d476055947fd3e64` |
| `configs/w2_llmseg_subsets.json` | `dfd53ce8e9a89d85ba828de6c0b3e331f83bfcf7d476cfc7ed1751e64c7497ae` |
| `third_party/lisa/requirements_lisa.txt` | `c8b7e202ddd02c6d5b2644f30c2ff9497649b6048df47d014217e733f74e5f2f` |
| `third_party/lisa/PROVENANCE.md` | `2ab8d4ec28e252504d5a15c882d28922ef1645c0f9d4a9594e3781c937ba6ad4` |
| `scripts/server/eval_lib.sh` | `5af0426385a751c1bb2941438215af4745d9c37f5a3a095ff7f0b0a5d185c835` |

官方文件逐个的 sha256 由 `src/lisa/common.check_vendor()` 在运行前对照 PROVENANCE.md 核对，不另列。

## 做法

1. hpcmgt 上 `squeue`。建工作树 wt0097（稀疏，含 `third_party/lisa/`、`src/`、`scripts/`、`configs/`、`tests/_lisa_fakes.py`），核对上表 11 个 sha256。
   - 确认 Lustre 余量 ≥ 19 GB。只记录，D95。
2. **登录节点：建环境、下权重**（照 `third_party/lisa/PROVENANCE.md` 的"环境"一节）：
   ```bash
   VI=/home/mail/chenyuma/visint
   $VI/env/bin/python -m venv --system-site-packages $VI/lisa/venv
   $VI/lisa/venv/bin/python -m pip install --no-cache-dir --only-binary :all: --no-deps -r $VI/wt/wt0097/third_party/lisa/requirements_lisa.txt --dry-run
   # 演练输出只应出现 requirements_lisa.txt 的 24 个包；没问题再去掉 --dry-run 正式装
   $VI/lisa/venv/bin/python -m pip freeze --path $VI/lisa/venv/lib/python3.12/site-packages   # 贴进报告
   cd $VI/wt/wt0097 && HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_DISABLE_XET=1 $VI/env/bin/python scripts/lisa_w2.py weights --download
   HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub $VI/env/bin/python scripts/lisa_w2.py weights --verify
   ```
   - 下载慢或连不上时可加 `HF_ENDPOINT=https://hf-mirror.com`。pip 不留缓存，不碰 `$VI/env` 与别人的目录。
   - **import 冒烟**：在登录节点上用 venv 的 python 跑 `scripts/run_lisa.py --help`，再按 PROVENANCE 的冒烟命令 import 官方模块，确认不报错。
     - 如果 pycocotools 在 numpy 1.26.4 下 import 失败，照 PROVENANCE 的备选往 venv 加 `pycocotools==2.0.8`，在报告里写明，不改其他。
3. **06**（第 2 步都通过后再发；06 空闲）：
   ```bash
   ssh -n hpcgpu06 'setsid nohup srun --jobid=157296 --overlap -n1 -c 8 \
     env PATH=/home/mail/chenyuma/visint/env/bin:/usr/bin:/bin TINY=0 EV_CAP_H=2 \
     bash /home/mail/chenyuma/visint/wt/wt0097/prompts/server_hpc/0097-hpc-06-lisa/lisa.sh \
     > /home/mail/chenyuma/visint/logs/eval_0097.driver.log 2>&1 < /dev/null &'
   ```
   - 驱动依次做：核对 venv 与权重 → claim_gpu → 导出清单 → 每个划分试跑 20 句 → 续跑全量 → 计分。
   - 心跳在 `eval_0097.status`。推送循环 TAG 0097，路径 `experiments/W2/0097_lisa_hpcgpu06`。
4. 试跑一步跑完后先报 `raw/logs/` 里的加载用时、每句用时、显存峰值。
5. 跑完贴每个划分的 cIoU、gIoU、failures，以及 `n_exif_rotated`、`n_image_hw_mismatch`。写 `reports/0097-hpc-06-lisa.md`，再跑一次 `squeue`。

管理员在开发机上用 tiny 小数据加替身运行器彩排过：首次运行与驱动重启两种情形 rc 都是 0，叶目录能被 `w2_test_table.preds` 读出；目录名不带运行编号或不含 `_lisa` 时拒绝。
