| 项 | 值 |
|---|---|
| 加载类 / processor / image processor / tokenizer | Gemma4ForConditionalGeneration / Gemma4Processor / Gemma4ImageProcessor / GemmaTokenizer |
| attn_implementation | sdpa |
| LLM 层数 | 35 |
| 层类型计数 | full_attention=7, sliding_attention=28 |
| 全注意力层下标（0 起） | [4, 9, 14, 19, 24, 29, 34] |
| hidden size / intermediate | 1536 / 6144 |
| Q 头 / KV 头 | 8 / 1 |
| head_dim（按层类型） | full_attention=512, sliding_attention=256 |
| sliding_window | 512 |
| num_kv_shared_layers | 20：第 15–34 层不带 k_proj/v_proj，复用 KV；来源层（store_full_length_kv）：{'sliding_attention': 13, 'full_attention': 14} |
| MLP intermediate（非共享 / 共享层） | 非共享=6144, 共享=12288（use_double_wide_mlp=True） |
| hidden_size_per_layer_input（PLE） | 256；embed_tokens_per_layer 形状 [262144, 8960] |
| use_bidirectional_attention | None |
| final_logit_softcapping | 30.0 |
| rope | {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"}, "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}} |
| 视觉编码器 层 / hidden / heads | 16 / 768 / 12 |
| patch_size / pooling_kernel_size | 16 / 3（每个 LLM 视觉 token = 缩放后图像的 48×48 px） |
| 参数量（加载后） | total=5,104,297,504, language_model=4,628,569,344, vision_tower=167,364,608, audio_tower=304,824,608, embed_vision=1,179,648, embed_audio=2,359,296 |
| config vocab / 嵌入行 / lm_head 行 | 262144 / 262144 / 262144 |
| len(tokenizer) / 基础词表 | 262144 / 262144 |
| 嵌入行 − len(tokenizer) | 0 |
| lm_head 与嵌入同一存储 | True |
| 嵌入类 / embed_scale | Gemma4TextScaledWordEmbedding / tensor(39.2500, device='cuda:0', dtype=torch.bfloat16) |
| 嵌入：基础词表随机 2 万行 | n=20000，范数 0.8757–1.5417（均值 1.1936），全零行 0 |
| 嵌入：超出 tokenizer 的行 | — |
| `<unusedN>` 一类占位 token | 6227 个，id 6–262143（不连续），首 [['<unused0>', 6], ['<unused1>', 7], ['<unused2>', 8]]，末 [['<unused6225>', 262142], ['<unused6226>', 262143]]；嵌入 n=6227，范数 0.8757–1.4897（均值 1.0344），全零行 0 |
| output_hidden_states | 36 个张量（= 层数+1：True），形状全同：True，例 [1, 284, 1536] |