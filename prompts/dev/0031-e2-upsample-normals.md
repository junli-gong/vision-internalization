[DEV-TASK 0031] E2：双线性上采样改为可分离矩阵乘（semseg 提速约 25 倍，定义不变）+ 撤掉法向许可门 + 估算覆盖法向与 pre_proj

## 背景

- RUN-0026（A800）的 E2 估算：semseg 一步 12–32 秒，depth 与 refseg 约 1 秒。8 个模型合计 179 GPU 小时，semseg 占 80%（`reports/0026-a800-delta.md` 第 5 节）。
- 原因已在 B200 上用微基准查明（D63-3）：
  - 测的是 2 个探针、150 类、特征网格 16×21、上采样到 512×683，再加交叉熵；
  - `F.interpolate(mode="bilinear", align_corners=False)` 的前向加反传每块 122 ms；
  - 同一个双线性插值写成 `A_h · X · A_wᵀ`（行、列各一个插值矩阵），每块 0.8 ms，前向最大差 6.7e-6；
  - 瓶颈是 `F.interpolate` 反传的原子加。
- 法向数据许可已确认（D59-6），许可门可以撤掉（D63-4）。
- E2 还没有正式运行，配置 sha256 变化无妨，写进总结。

先读：
- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- `DECISIONS.md` 的 D53、D59、D60、D62、D63
- `src/e2/`（尤其 `tasks.upsample`、`runner.train_step` 与 `collect`、`estimate.py`）、`scripts/run_e2.py`、`configs/e2_probes.json`
- `reports/0026-a800-delta.md`

## 任务

### 1. 可分离矩阵乘的双线性上采样（`src/e2/tasks.upsample`）

- 用两个插值矩阵实现：`A_h [H, h]`、`A_w [W, w]`，`out = A_h · X · A_wᵀ`，对 `[P, C, h, w]` 批量做。
- 插值矩阵按 PyTorch 的公式构造，与 `F.interpolate(mode="bilinear", antialias=False)` 一致：
  - `align_corners` 取 False 与 True 两种都要支持；
  - 源坐标的计算、边界截断按 PyTorch 源码。
- 插值矩阵按 `(n_in, n_out, align_corners, device, dtype)` 缓存。
- **数值精度**：
  - 矩阵乘必须是真正的 fp32，不能用 TF32；
  - 在函数里显式保证这一点，不依赖全局设置，写明用的是哪种机制（torch 2.13 的 API）；
  - 加一条测试：故意打开全局 TF32，结果仍与关闭时逐位相同，或在容差 1e-6 内。CPU 上做不到的话说明理由。
- **等价性测试**（CPU，fp32 与 float64）：与 `F.interpolate` 比较前向输出与对输入的梯度。
  - 形状：上采样、下采样、恒等、奇数尺寸、1 像素宽，以及 E2 实际出现的网格与 GT 尺寸，例如 16×21 → 512×683、15×20 → 480×640；
  - 容差：float64 下 ≤ 1e-12；fp32 下相对误差 ≤ 1e-5。
- 配置 `upsample` 加 `"impl": "separable_matmul"`，并写 basis：D63、RUN-0026 的微基准。保留 `"interpolate"` 作为可选实现，用于对照测试。
- 训练与评测共用同一个函数。
- 全仓库搜一遍其他**训练时带反传**的 `F.interpolate` 双线性调用（例如 E3a 的解码器、`src/executors/`），**只列出来，不改**，写进总结。

### 2. 撤掉法向许可门（D59-6、D63-4）

- `scripts/run_e2.py`：真模型跑 normals 不再需要 `--enable-normals`。这个参数保留为无作用的兼容参数，帮助文本注明 D59-6。
- `configs/e2_probes.json` 的 `normals.license_gate` 改为"D59-6：许可已确认（Bae 2021 / GeoNet）"。
- 相应的测试改为：不加参数也能跑 normals。

### 3. 估算覆盖全部任务与参考线

- `--estimate --task all` 包含法向。
- 给了 `--reference pre_proj` 时，估算要把 pre_proj 那组探针计入：前向开销、探针数、步时间。
- 估算报告的表头写明：层、参考线、是否含法向。
- 输出加一列"上采样实现"。

### 4. 可选（便宜才做）

- 不改变任何数值的工程优化可以顺手做，但每一项都要写明，并用测试证明数值不变。例如上采样之后的交叉熵少一次 `.contiguous()` 拷贝。
- 不要改 `probe_chunk` 与 `probe_chunk_max_elements` 的值。

## 允许改动的文件

- `src/e2/`、`scripts/run_e2.py`、`configs/e2_probes.json`
- E2 相关的测试与 fixture

**不许碰**：`src/adapters/`、`src/e1/`、`src/e3/`、`src/e5/`、`src/synth/`、`src/data/`、`configs/e1_prompts.json`、`configs/e3b_train.json`、`configs/e5_prompts.json`。另有 DEV-0030 在并行改 `src/e3/` 与 `scripts/run_e1.py`，不要动它们。

## 环境与验收

- conda 环境 `visint`，不装包，不下载权重，不联网；测试加 `HF_HUB_OFFLINE=1`。
- 验收：
  - `conda run -n visint pytest tests/ -q`
  - `conda run -n visint bash scripts/pytest_server_parity.sh`
  - `run_e2.py --tiny`，全部任务，含 normals 与 pre_proj
  - `run_e2.py --estimate --tiny --task all --reference post_proj pre_proj`
- 总结里写明：
  - 插值矩阵的构造与 PyTorch 源码的对应；
  - fp32 的保证机制；
  - 等价性测试的最大误差；
  - E2 配置的新 sha256；
  - CPU 上新旧实现的速度对比，仅供参考；
  - 其他训练时带反传的双线性调用清单。
