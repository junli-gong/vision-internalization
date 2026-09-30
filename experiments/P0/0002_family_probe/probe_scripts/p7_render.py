"""RUN-0002: render JSON outputs of one family into markdown fragments (all numbers read from JSON).
usage: python p7_render.py <tag>   -> $PROJ/logs/0002/<tag>/render_*.md + render.json
"""
import os, sys, json
tag = sys.argv[1]
P = os.path.join(os.environ['PROJ'], 'logs/0002', tag)
J = lambda f: json.load(open(os.path.join(P, f)))
esc = lambda x: str(x).replace('|', '\\|')
F = J('facts.json'); fam = F['family']
out = {}

# ---- structure table
s = ['| 项 | 值 |', '|---|---|']
def row(k, v): s.append(f'| {k} | {esc(v)} |')
row('加载类 / processor / image processor / tokenizer', f"{F['classes']['model']} / {F['classes']['processor']} / {F['classes']['image_processor']} / {F['classes']['tokenizer']}")
row('attn_implementation', F['attn_implementation'])
row('LLM 层数', F['n_layers'])
row('层类型计数', ', '.join(f'{k}={v}' for k, v in sorted(F['layer_type_counts'].items())))
if F['full_attention_layer_idx'] and len(F['full_attention_layer_idx']) < F['n_layers']:
    row('全注意力层下标（0 起）', F['full_attention_layer_idx'])
tc = F['text_cfg']
row('hidden size / intermediate', f"{tc['hidden_size']} / {tc['intermediate_size']}")
row('Q 头 / KV 头', f"{tc['num_attention_heads']} / {tc['num_key_value_heads']}")
if fam == 'gemma':
    hd = sorted(set((l['layer_type'], l['head_dim']) for l in F['layers']))
    row('head_dim（按层类型）', ', '.join(f'{a}={b}' for a, b in hd))
    row('sliding_window', tc['sliding_window'])
    kv = F['kv_shared']
    row('num_kv_shared_layers', f"{tc['num_kv_shared_layers']}：第 {kv['kv_shared_layers'][0]}–{kv['kv_shared_layers'][-1]} 层不带 k_proj/v_proj，复用 KV；来源层（store_full_length_kv）：{kv['kv_source_layers(store_full_length_kv)']}")
    mi = sorted(set((l['is_kv_shared_layer'], l['mlp_intermediate']) for l in F['layers']))
    row('MLP intermediate（非共享 / 共享层）', ', '.join(f"{'共享' if a else '非共享'}={b}" for a, b in mi) + f"（use_double_wide_mlp={tc['use_double_wide_mlp']}）")
    row('hidden_size_per_layer_input（PLE）', f"{tc['hidden_size_per_layer_input']}；embed_tokens_per_layer 形状 {F['vocab'].get('embed_tokens_per_layer_shape')}")
    row('use_bidirectional_attention', tc['use_bidirectional_attention'])
    row('final_logit_softcapping', tc['final_logit_softcapping'])
    row('rope', json.dumps(tc['rope_parameters']))
else:
    row('head_dim', tc['head_dim'])
    row('rope', json.dumps(tc['rope_parameters']))
    row('max_position_embeddings', tc['max_position_embeddings'])
v = F['vision']
if fam == 'gemma':
    row('视觉编码器 层 / hidden / heads', f"{v['num_hidden_layers']} / {v['hidden_size']} / {v['num_attention_heads']}")
    row('patch_size / pooling_kernel_size', f"{v['patch_size']} / {v['pooling_kernel_size']}（每个 LLM 视觉 token = 缩放后图像的 {v['px_per_llm_token_side']}×{v['px_per_llm_token_side']} px）")
else:
    row('视觉编码器 层 / hidden / heads', f"{v['num_hidden_layers']} / {v['hidden_size']} / {v['num_attention_heads']}")
    row('patch_size / tile / downsample_ratio', f"{v['patch_size']} / {v['image_size']} / {v['downsample_ratio']}（每 tile {v['tokens_per_tile']} token；每 token = tile 的 {v['px_per_llm_token_side']}×{v['px_per_llm_token_side']} px）")
    row('vision_feature_layer / projector act', f"{v['vision_feature_layer']} / {v['projector_hidden_act']}")
row('参数量（加载后）', ', '.join(f'{k}={v_:,}' for k, v_ in F['param_counts'].items()))
vo = F['vocab']
row('config vocab / 嵌入行 / lm_head 行', f"{vo['config_vocab_size']} / {vo['embed_rows']} / {vo['lm_head_rows']}")
row('len(tokenizer) / 基础词表', f"{vo['len_tokenizer']} / {vo['tokenizer_vocab_size_base']}")
row('嵌入行 − len(tokenizer)', vo['n_rows_beyond_tokenizer'])
row('lm_head 与嵌入同一存储', vo['lm_head_is_embed(same storage)'])
if vo.get('embed_scale') is not None:
    row('嵌入类 / embed_scale', f"{vo['embed_class']} / {vo['embed_scale']}")
def rs(d):
    return '—' if d is None else f"n={d['n']}，范数 {d['norm_min']:.4f}–{d['norm_max']:.4f}（均值 {d['norm_mean']:.4f}），全零行 {d['n_exact_zero_rows']}"
row('嵌入：基础词表随机 2 万行', rs(F['embed_row_stats']['base_vocab_sample20k']))
row('嵌入：超出 tokenizer 的行', rs(F['embed_row_stats']['beyond_tokenizer_rows']))
u = F['unused_like_tokens']
row('`<unusedN>` 一类占位 token', f"{u['n']} 个" + (f"，id {u['id_range'][0]}–{u['id_range'][1]}（不连续），首 {u['first5'][:3]}，末 {u['last5'][-2:]}；嵌入 {rs(u.get('embed_row_stats'))}" if u['n'] else ''))
row('output_hidden_states', f"{F['output_hidden_states']['n']} 个张量（= 层数+1：{F['output_hidden_states']['n'] == F['output_hidden_states']['n_layers+1']}），形状全同：{F['output_hidden_states']['all_same_shape']}，例 {F['output_hidden_states']['shapes'][0]}")
out['structure'] = '\n'.join(s)

# ---- added tokens
s = ['| id | token | special | 出现在 chat template 里 | 嵌入范数 |', '|---|---|---|---|---|']
for t in F['added_special_tokens(non-unused)']:
    en = '' if t['embed_norm'] is None else f"{t['embed_norm']:.4f}"
    s.append(f"| {t['id']} | `{esc(t['content'])}` | {t['special']} | {t['in_chat_template']} | {en} |")
out['added_tokens'] = '\n'.join(s)

# ---- new token
n = F['new_token_test']; rz = F['resize_token_embeddings(len(tok_with_2_new))']
s = [f"- `add_tokens(['<g0>'])` 返回 {n['n_added_normal']}，`add_tokens(['<g1>'], special_tokens=True)` 返回 {n['n_added_special']}；id(`<g0>`)={n['id_<g0>']}，id(`<g1>`)={n['id_<g1>']}；len(tokenizer) {n['len_before']}→{n['len_after']}；新 id < 嵌入行数：{n['new_id_lt_embed_rows']}" + (f"；id {n['id_<g0>']} 原有嵌入范数 {n['embed_row_at_<g0>_norm']:.4f}" if n['embed_row_at_<g0>_norm'] is not None else '') + '。',
     f"- `resize_token_embeddings(len(tok))`：嵌入 {rz.get('rows_before')}→{rz.get('rows_after')}，lm_head→{rz.get('lm_head_rows_after')}，config vocab→{rz.get('config_vocab_after')}" + (f"；PLE `embed_tokens_per_layer` {rz['ple_before']}→{rz['ple_after']}" if rz.get('ple_before') else '') + '。',
     '', '| 输入 | ids | tokens |', '|---|---|---|']
for k, t in n['tokenization'].items():
    s.append(f"| `{k!r}` | {t['ids']} | {esc(t['tokens'])} |")
out['new_token'] = '\n'.join(s)

# ---- templates
s = ['| 调用方式 | 生成提示（原样，截取末段） | 备注 |', '|---|---|---|']
for k, r in F['chat_template_renders'].items():
    txt = r.get('text', r.get('error', ''))
    tail = txt if len(txt) < 160 else '…' + txt[-160:]
    note = r.get('log', '').strip()
    s.append(f"| {esc(k)} | `{esc(repr(tail))}` | {esc(note)} |")
out['templates'] = '\n'.join(s)
out['full_off_render'] = F['chat_template_renders']['enable_thinking=False | image' if fam == 'gemma' else 'default(no kwarg) | image']['text']
out['genconfig'] = F['generation_config']; out['eos'] = F['eos_tokens']; out['config_eos'] = F['config_eos']
out['template_variables'] = F['template_variables']

# ---- image tokens
if fam == 'gemma':
    s = ['| 输入（W×H）与参数 | 视觉 token 数 | 缩放后 (h, w) px | LLM 网格 (h, w) | pixel_values 形状（含 -1 填充的 patch） | 有效 patch / 填充 patch |', '|---|---|---|---|---|---|']
    for k, d in F['image_tokenization'].items():
        if 'error' in d:
            s.append(f'| {k} | ERROR {esc(d["error"])} | | | | |'); continue
        s.append(f"| {k} | {d['n_image_tokens']} | {tuple(d['resized_px_(a,b)'])} | {tuple(d['llm_grid_(a,b)'])} | {d['pixel_values_shape']} | {d['n_valid_patches']} / {d['n_padded_patches']} |")
else:
    s = ['| 输入（W×H）与参数 | 视觉 token 数 | tile 数（pixel_values 第 0 维） | pixel_values 形状 |', '|---|---|---|---|']
    for k, d in F['image_tokenization'].items():
        if 'error' in d:
            s.append(f'| {k} | ERROR {esc(d["error"])} | | |'); continue
        s.append(f"| {k} | {d['n_image_tokens']} | {d['n_tiles']} | {d['pixel_values_shape']} |")
out['image_tokens'] = '\n'.join(s)
out['image_processor_attrs'] = F['image_processor_attrs']; out['processor_defaults'] = F['processor_default_images_kwargs']
out['two_images'] = F['two_images']

# ---- positions
p = F['positions']
s = [f"position_ids 在 rotary 处的形状 {p['position_ids_shape_at_rotary']}，等于 0..L-1：{p['position_ids_is_arange']}（seq_len {p['seq_len']}）。", '',
     '| 段 | token 下标 | 个数 | 开头 token | 前 5 个 position | 后 3 个 position |', '|---|---|---|---|---|---|']
for g in p['segments']:
    s.append(f"| {g['type']} | {g['tok_range'][0]}–{g['tok_range'][1]} | {g['n']} | {esc(g['first_tokens'])} | {g['pos_first5']} | {g['pos_last3']} |")
if 'vision_encoder_image_position_ids' in p:
    vp = p['vision_encoder_image_position_ids']
    s += ['', f"视觉编码器的 `image_position_ids`（processor 输出，二维 patch 坐标）：形状 {vp['shape']}，有效 {vp['n_valid']}、填充(-1) {vp['n_pad(-1)']}；前 6 个 {vp['first6']}，末 3 个 {vp['last3']}；第 0 维最大 {vp['dim0_max']}，第 1 维最大 {vp['dim1_max']}。"]
out['positions'] = '\n'.join(s); out['positions_text'] = p['example_text']

# ---- box
B = J('box_zeroshot.json')
s = [f"提示词 `{B['prompt_template']}`，出处：{B['prompt_source']}；生成设置 {json.dumps(B['generation'], ensure_ascii=False)}。", '']
for c in B['cases']:
    s.append(f"- `{c['image']}`（{c['image_wh'][0]}×{c['image_wh'][1]}，视觉 token {c['n_image_tokens']}" + (f"，tile {c['n_tiles']}" if 'n_tiles' in c else '') + f"），目标 \"{c['object']}\"，生成 {c['n_new_tokens']} token：`{c['raw_output_with_special']!r}`")
out['box'] = '\n'.join(s)

# ---- taskmod
T = J('taskmod_bf16.json')
s = [f"指令 A=`{T['instructions'][0]}`，B=`{T['instructions'][1]}`（token 长度 {T['instr_token_lens']}），bf16，attn={T['attn_implementation']}，图像 token {T['image_first']['n_image_tokens']} 个，seq_len {T['image_first']['seq_len']}。", '',
     '| 层 | 图在前：torch.equal | 图在前：max\\|Δ\\| | 指令在前：torch.equal | 指令在前：max\\|Δ\\| | 指令在前：cos（展平） | 指令在前：逐 token cos 最小 | 指令在前：变化 token 数 |', '|---|---|---|---|---|---|---|---|']
for a, b in zip(T['image_first']['layers'], T['instruction_first']['layers']):
    s.append(f"| {a['layer']} | {a['equal']} | {a['max_abs_delta']:.4g} | {b['equal']} | {b['max_abs_delta']:.4g} | {b['cos_flat']:.6f} | {b['cos_min_token']:.6f} | {b['n_tokens_changed']} |")
s += ['', f"图在前：{sum(r['equal'] for r in T['image_first']['layers'])}/{len(T['image_first']['layers'])} 层逐位相同；指令在前：{sum(r['equal'] for r in T['instruction_first']['layers'])}/{len(T['instruction_first']['layers'])} 层逐位相同。"]
out['taskmod'] = '\n'.join(s)
out['taskmod_summary'] = {'image_first_equal': [sum(r['equal'] for r in T['image_first']['layers']), len(T['image_first']['layers'])],
                          'instr_first_equal': [sum(r['equal'] for r in T['instruction_first']['layers']), len(T['instruction_first']['layers'])],
                          'last_cos_flat': T['instruction_first']['layers'][-1]['cos_flat'], 'last_cos_min': T['instruction_first']['layers'][-1]['cos_min_token']}

# ---- mem
I = J('mem_infer.json'); L = J('mem_lora.json')
s = ['| 测量 | batch | 峰值 allocated GiB | 峰值 reserved GiB | 备注 |', '|---|---|---|---|---|',
     f"| 仅加载权重（bf16） | — | {I['after_load']['alloc_GiB']} | {I['after_load']['peak_reserved_GiB']} | |"]
for bs in [1, 16]:
    for nm in ['logits_to_keep=1', 'full_logits']:
        d = I[f'forward_bs{bs}_{nm}']
        s.append(f"| 推理前向，{I['n_visual_tokens']} 视觉 token（seq {I['seq_len']}，{json.dumps(I['image_kwargs'])}），{nm} | {bs} | {d['peak_alloc_GiB']} | {d['peak_reserved_GiB']} | logits {d['logits_shape']}；墙钟 {d['wall_s']} s |")
for rep in [0, 1]:
    d = L[f'lora_r32_lm_linear_bs8_gc_rep{rep}']
    s.append(f"| LoRA r=32（language_model 内全部 nn.Linear），梯度检查点，前向+反向第 {rep + 1} 次，全序列 LM loss（seq {L['seq_len']}） | 8 | {d['peak_alloc_GiB']} | {d['peak_reserved_GiB']} | 可训练参数 {d['trainable_params']:,}；LoRA 模块 {d['n_lora_modules']}；墙钟 {d['wall_s_fwd_bwd']} s；loss {d['loss']:.6f} |")
g = I['generate_bs16_max_new_64']; r = g['runs']
s.append(f"| HF generate，greedy，思考关，max_new_tokens=64 | 16 | {max(x['peak_alloc_GiB'] for x in r)} | {max(x['peak_reserved_GiB'] for x in r)} | 3 次：{', '.join(str(x['samples_per_s']) for x in r)} 样本/s（墙钟 {', '.join(str(x['wall_s']) for x in r)} s）；new_tokens 矩阵 {r[0]['new_tokens_matrix']}；prompt {g['prompt_len']} token |")
out['mem'] = '\n'.join(s)
out['lora_targets'] = L['lora_target_suffix_counts']
json.dump(out, open(os.path.join(P, 'render.json'), 'w'), indent=1, ensure_ascii=False, default=str)
for k, v_ in out.items():
    if isinstance(v_, str):
        open(os.path.join(P, f'render_{k}.md'), 'w').write(v_)
print('ok', fam, list(out))
