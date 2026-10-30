# 统一主表：与预注册 U-4 的对照（管理员；`unified_table.md` 由脚本重写，对照写在这里）


- **LISA-7B**（RUN-0097，`xinlai/LISA-7B-v1`，官方 refer_seg 协议）：合并 cIoU 0.612 [0.590, 0.635]，gIoU 0.613；X-2 − LISA +0.111 [+0.088, +0.133]。
  - 预测 cIoU 0.62–0.74、中心 0.68：**略低于范围**。
  - 预测 X-2 − LISA 为 −0.04 到 +0.08：**高于范围**。
  - 预测 CI 下界 > 0 的概率约 45%：发生了。
  - 排序：预测"LISA ≈ W2-c"，实际 LISA 比 W2-c 低约 0.03。
  - 可能的原因，不作因果主张：
    - 发布的 v1 权重不是论文在 refCOCO(+/g) 上微调的版本。微调版在这 5 个划分的完整 test 上平均约 0.70（`docs/references/refseg_sota_notes.md`）。
    - refcoco+ testB 只有 0.483，空 mask 11 句。
  - 1,500 句没有一句失败；EXIF 旋转与尺寸不符都是 0。
