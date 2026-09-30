| 项 | 值 |
|---|---|
| 加载类 / processor / image processor / tokenizer | InternVLForConditionalGeneration / InternVLProcessor / GotOcr2ImageProcessor / Qwen2Tokenizer |
| attn_implementation | sdpa |
| LLM 层数 | 28 |
| 层类型计数 | full_attention=28 |
| hidden size / intermediate | 2048 / 6144 |
| Q 头 / KV 头 | 16 / 8 |
| head_dim | 128 |
| rope | {"rope_theta": 1000000, "rope_type": "default"} |
| max_position_embeddings | 40960 |
| 视觉编码器 层 / hidden / heads | 24 / 1024 / 16 |
| patch_size / tile / downsample_ratio | [14, 14] / [448, 448] / 0.5（每 tile 256 token；每 token = tile 的 28×28 px） |
| vision_feature_layer / projector act | -1 / gelu |
| 参数量（加载后） | total=2,348,347,392, language_model=1,720,574,976, vision_tower=304,012,288, multi_modal_projector=12,595,200 |
| config vocab / 嵌入行 / lm_head 行 | 151936 / 151936 / 151936 |
| len(tokenizer) / 基础词表 | 151679 / 151643 |
| 嵌入行 − len(tokenizer) | 257 |
| lm_head 与嵌入同一存储 | False |
| 嵌入：基础词表随机 2 万行 | n=20000，范数 0.3692–2.2132（均值 1.4829），全零行 0 |
| 嵌入：超出 tokenizer 的行 | n=257，范数 0.3317–0.3700（均值 0.3509），全零行 0 |
| `<unusedN>` 一类占位 token | 0 个 |
| output_hidden_states | 29 个张量（= 层数+1：True），形状全同：True，例 [1, 3346, 2048] |