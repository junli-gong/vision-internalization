"""Render §5 JSON outputs into markdown fragments (all numbers read from JSON). Writes $PROJ/logs/0001/p5/render_*.md"""
import json, os
P = os.path.join(os.environ['PROJ'], 'logs/0001/p5')
J = lambda *a: json.load(open(os.path.join(P, *a)))
F = {t: J(t, 'facts.json') for t in ['qwen3.5-0.8b', 'qwen3.5-9b']}
out = {}
esc = lambda x: str(x).replace('|', '\\|')  # GFM table cells

# ---- 5.1 structure
s = ['| 项 | Qwen3.5-0.8B | Qwen3.5-9B |', '|---|---|---|']
def row(name, f):
    s.append(f'| {name} | ' + ' | '.join(str(f(F[t])) for t in F) + ' |')
row('加载类', lambda R: R['model_class'])
row('processor / image processor / tokenizer', lambda R: f"{R['processor_class']} / {R['image_processor_class']} / {R['tokenizer_class']}")
row('LLM 层数', lambda R: R['n_layers'])
row('层类型计数', lambda R: ', '.join(f'{k}={v}' for k, v in sorted(R['layer_type_counts'].items())))
row('全注意力层下标（0 起）', lambda R: R['full_attention_layer_idx'])
row('LLM hidden size', lambda R: R['text_hidden_size'])
row('全注意力：Q 头 / KV 头 / head_dim / 输出门', lambda R: f"{R['full_attn']['num_attention_heads']} / {R['full_attn']['num_key_value_heads']} / {R['full_attn']['head_dim']} / {R['full_attn']['attn_output_gate']}")
row('Gated DeltaNet：K 头 / V 头 / k_dim / v_dim / conv kernel', lambda R: f"{R['gdn']['linear_num_key_heads']} / {R['gdn']['linear_num_value_heads']} / {R['gdn']['linear_key_head_dim']} / {R['gdn']['linear_value_head_dim']} / {R['gdn']['linear_conv_kernel_dim']}")
row('视觉编码器 depth / hidden / heads', lambda R: f"{R['vision']['depth']} / {R['vision']['hidden_size']} / {R['vision']['num_heads']}")
row('patch_size / spatial_merge_size / temporal_patch_size', lambda R: f"{R['vision']['patch_size']} / {R['vision']['spatial_merge_size']} / {R['vision']['temporal_patch_size']}")
row('每个 LLM 视觉 token 对应（缩放后图像）', lambda R: f"{R['vision']['pixels_per_llm_token_side']}×{R['vision']['pixels_per_llm_token_side']} px")
row('deepstack_visual_indexes', lambda R: R['vision']['deepstack_visual_indexes'])
row('参数量：总 / 视觉 / 语言模型（加载后）', lambda R: f"{R['param_counts']['total']:,} / {R['param_counts']['visual']:,} / {R['param_counts']['language_model']:,}")
row('config vocab_size / 嵌入行数 / lm_head 行数', lambda R: f"{R['vocab']['config_vocab_size']} / {R['vocab']['embed_rows']} / {R['vocab']['lm_head_rows']}")
row('len(tokenizer) / 基础词表（不含 added）', lambda R: f"{R['vocab']['len_tokenizer']} / {R['vocab']['tokenizer_vocab_size_base']}")
row('嵌入行数 − len(tokenizer)（超出 tokenizer 的行）', lambda R: R['vocab']['n_rows_beyond_tokenizer'])
row('tie_word_embeddings / lm_head 与嵌入同一存储', lambda R: f"{R['vocab']['tie_word_embeddings(top)']} / {R['vocab']['lm_head_is_embed (same storage)']}")
def rs(d):
    return f"范数 {d['norm_min']:.4f}–{d['norm_max']:.4f}（均值 {d['norm_mean']:.4f}），全零行 {d['n_exact_zero_rows']}，与首行最大差 {d['max_abs_diff_to_first_row']:.4g}"
row('嵌入：超出 tokenizer 的 243 行', lambda R: rs(R['embed_row_stats']['beyond_tokenizer_rows']))
row('嵌入：基础词表随机 2 万行', lambda R: rs(R['embed_row_stats']['base_vocab_sample20k']))
row('lm_head：超出 tokenizer 的 243 行', lambda R: rs(R['lm_head_row_stats']['beyond_tokenizer_rows']) if 'lm_head_row_stats' in R else '（与嵌入共享）')
row('lm_head：基础词表随机 2 万行', lambda R: rs(R['lm_head_row_stats']['base_vocab_sample20k']) if 'lm_head_row_stats' in R else '（与嵌入共享）')
out['5_1_structure'] = '\n'.join(s)

# added tokens table
s = ['| id | token | special | 出现在 chat template 里 | 0.8B 嵌入范数 | 9B 嵌入范数 | 9B lm_head 范数 |', '|---|---|---|---|---|---|---|']
a8 = {t['id']: t for t in F['qwen3.5-0.8b']['added_tokens']}; a9 = {t['id']: t for t in F['qwen3.5-9b']['added_tokens']}
for i in sorted(a8):
    s.append(f"| {i} | `{esc(a8[i]['content'])}` | {a8[i]['special']} | {a8[i]['in_chat_template']} | {a8[i]['embed_norm']:.4f} | {a9[i]['embed_norm']:.4f} | {a9[i]['lm_head_norm']:.4f} |")
out['5_1_added_tokens'] = '\n'.join(s)

# new token
s = []
for t, R in F.items():
    n = R['new_token_test']
    s.append(f"- **{t}**：`add_tokens(['<g0>'])` 返回 {n['n_added_normal']}，`add_tokens(['<g1>'], special_tokens=True)` 返回 {n['n_added_special']}；"
             f"id(`<g0>`)={n['id_<g0>']}，id(`<g1>`)={n['id_<g1>']}，len(tokenizer) {n['len_before']}→{n['len_after']}；新 id < 嵌入行数：{n['new_id_lt_embed_rows']}；"
             f"id {n['id_<g0>']} 这一行原有嵌入的范数 {n['embed_row_at_<g0>_norm']:.4f}。"
             f"`resize_token_embeddings(len(tok))` 后行数：{R['resize_token_embeddings(len(tok_with_2_new))']['rows_before']}→{R['resize_token_embeddings(len(tok_with_2_new))']['rows_after']}"
             f"（lm_head {R['resize_token_embeddings(len(tok_with_2_new))']['lm_head_rows_after']}）。")
s.append('')
s.append('分词结果（0.8B 与 9B 一致）：')
s.append('')
s.append('| 输入 | ids | tokens |'); s.append('|---|---|---|')
for k, v in F['qwen3.5-0.8b']['new_token_test']['tokenization'].items():
    same = v['ids'] == F['qwen3.5-9b']['new_token_test']['tokenization'][k]['ids']
    s.append(f"| `{k!r}` | {v['ids']} | {v['tokens']}{'' if same else ' (9B 不同!)'} |")
out['5_1_new_token'] = '\n'.join(s)

# ---- 5.2 templates
s = []
for t, R in F.items():
    s.append(f'**{t}**')
    s.append('')
    s.append('| 调用方式 | 生成提示结尾（原样） | 备注 |'); s.append('|---|---|---|')
    for k, v in R['chat_template_renders'].items():
        if not k.endswith('| text') and 'tokenizer' not in k:
            continue
        tail = v.get('text', v.get('error', ''))
        tail = tail[tail.rfind('<|im_start|>assistant'):]
        note = ('日志：' + v['log'].strip()) if v.get('log', '').strip() else ''
        s.append(f"| {esc(k.replace(' | ', ' · '))} | `{esc(repr(tail))}` | {esc(note)} |")
    s.append('')
out['5_2_templates'] = '\n'.join(s)
out['5_2_full_off_render'] = {t: R['chat_template_renders']['processor | enable_thinking=False | image']['text'] for t, R in F.items()}
out['5_2_genconfig'] = {t: {k: v for k, v in R['generation_config'].items() if v is not None} for t, R in F.items()}

# ---- 5.3 image tokens
s = ['| 输入（W×H）与参数 | image_grid_thw（patch 网格 t,h,w） | LLM 网格 (h, w) | 视觉 token 数 | 缩放后像素 (h, w) | 每 token 像素（缩放后） | 每 token 像素（原图） |', '|---|---|---|---|---|---|---|']
for k, v in F['qwen3.5-0.8b']['image_tokenization'].items():
    v9 = F['qwen3.5-9b']['image_tokenization'][k]
    same = v['image_grid_thw(patches)'] == v9['image_grid_thw(patches)']
    s.append(f"| {k} | {v['image_grid_thw(patches)']} | {tuple(v['llm_grid_hw'])} | {v['n_image_pad_tokens']} | {tuple(v['resized_hw_px'])} | {v['px_per_token_in_resized']:.0f} | {v['px_per_token_in_original']:.1f} |" + ('' if same else ' 9B 不同!'))
out['5_3_table'] = '\n'.join(s)
out['5_3_defaults'] = F['qwen3.5-0.8b']['image_processor_defaults']
out['5_3_two'] = F['qwen3.5-0.8b']['two_images']

# ---- 5.4 positions
s = []
for t, R in F.items():
    p = R['positions']
    s.append(f"**{t}**：`get_rope_index` 输出形状 {p['get_rope_index_shape']}，rope_deltas={p['rope_deltas']}；generate 内部构造的 position_ids 形状 {p.get('generate_position_ids_shape')}"
             f"（第 0 行是 0..L-1 的文本位置：{p.get('generate_row0_is_arange')}；第 1–3 行等于 get_rope_index：{p.get('generate_rows1to3_equal_get_rope_index')}）。图像 grid_thw={p['image_grid_thw']}，seq_len={p['seq_len']}。")
    s.append('')
    s.append('| 段 | token 下标 | 个数 | T 范围 | H 范围 | W 范围 | 前 5 个 (T,H,W) | 后 3 个 (T,H,W) |'); s.append('|---|---|---|---|---|---|---|---|')
    for g in p['segments']:
        s.append(f"| {g['type']} | {g['tok_range'][0]}–{g['tok_range'][1]} | {g['n']} | {g['T_range'][0]}–{g['T_range'][1]} | {g['H_range'][0]}–{g['H_range'][1]} | {g['W_range'][0]}–{g['W_range'][1]} | {g['first5_THW']} | {g['last3_THW']} |")
    s.append('')
out['5_4'] = '\n'.join(s)
out['5_4_text'] = F['qwen3.5-0.8b']['positions']['example_text']

# ---- 5.5
s = []
gt = J('coco_gt_2images.json')
for t in F:
    B = J(t, 'box_zeroshot.json')
    for c in B['cases']:
        s.append(f"- **{t}** `{c['image']}`（{c['image_wh'][0]}×{c['image_wh'][1]}，模型输入缩放为 h×w={c['resized_hw_px'][0]}×{c['resized_hw_px'][1]}），目标 \"{c['object']}\"，生成 {c['n_new_tokens']} token：`{c['raw_output_with_special']!r}`")
out['5_5_outputs'] = '\n'.join(s)
out['5_5_prompt'] = J('qwen3.5-0.8b', 'box_zeroshot.json')['prompt_template']
s = ['| 图 | 物体 | GT xyxy（像素） | GT xyxy ×1000/(W,H) |', '|---|---|---|---|']
for iid, g in gt.items():
    W, H = g['wh']
    for cname, b in g['objects']:
        if cname in ('cat', 'tv'):
            x1, y1, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
            s.append(f"| {g['file']} ({W}×{H}) | {cname} | [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] | [{x1/W*1000:.0f}, {y1/H*1000:.0f}, {x2/W*1000:.0f}, {y2/H*1000:.0f}] |")
out['5_5_gt'] = '\n'.join(s)

# ---- 5.6
K = J('gdn_kernels.json')
s = [f"形状 B={K['shapes']['B']}, T={K['shapes']['T']}（chunk=64，共 5 个 chunk，最后一个不满），H={K['shapes']['H']}, K=V={K['shapes']['K']}；fla {K['fla_version']}。", '']
for key, v in K['results'].items():
    s.append(f'**{key}**（输出 dtype：' + ', '.join(f'{n}={d.replace("torch.", "")}' for n, d in v['out_dtypes'].items()) + '）')
    s.append('')
    s.append('| 对比 | o max\\|Δ\\| | o mean\\|Δ\\| | o ‖Δ‖/‖ref‖ | o max rel | 末状态 S max\\|Δ\\| |'); s.append('|---|---|---|---|---|---|')
    for p, d in v['pairs'].items():
        fs = d.get('final_state', {}).get('max_abs')
        s.append(f"| {p} | {d['o']['max_abs']:.3e} | {d['o']['mean_abs']:.3e} | {d['o']['rel_fro']:.3e} | {d['o']['max_rel']:.3e} | {'' if fs is None else f'{fs:.3e}'} |")
    s.append('')
out['5_6_kernels'] = '\n'.join(s)
M = J('gdn_model_paths_inproc.json'); Mb = J('gdn_model_paths_blockfla.json')
s = ['| dtype | logits max\\|Δ\\| | mean\\|Δ\\| | ‖Δ‖/‖ref‖ | 逐位置 argmax 一致 | 末位置 argmax 一致 | 末位置 top5（fla） | 末位置 top5（torch） | 同路径重复两次逐位相同（fla / torch） | 进程内 torch 参考 与 屏蔽 fla 的独立进程 sha256 相同 |',
     '|---|---|---|---|---|---|---|---|---|---|']
for dn in ['bfloat16', 'float32']:
    d = M[f'{dn}_fla_vs_torchref']
    n = M['seq_len']
    agree = round(d['argmax_agree_frac'] * n)
    s.append(f"| {dn} | {d['max_abs']:.4g} | {d['mean_abs']:.4g} | {d['rel_fro']:.4g} | {agree}/{n} | {d['last_pos_argmax_equal']} | {M[f'{dn}_last_logits_top5_fla']} | {M[f'{dn}_last_logits_top5_torchref']} | {M[f'{dn}_fla_repeat_bitwise_equal']} / {M[f'{dn}_torchref_repeat_bitwise_equal']} | {M[f'{dn}_sha_torchref'] == Mb[f'{dn}_sha_fallback']} |")
out['5_6_model'] = '\n'.join(s)
out['5_6_paths'] = {'inproc_impl': M['impl'], 'blockfla_impl': Mb['impl'], 'inproc_bf16_log': M['bfloat16_log'], 'blockfla_bf16_log': Mb['bfloat16_log'],
                    'attn': M['bfloat16_attn_implementation'], 'seq_len': M['seq_len'], 'grid': M['image_grid_thw']}

# ---- 5.7
s = []
for t, dts in [('qwen3.5-0.8b', 'bf16'), ('qwen3.5-9b', 'bf16'), ('qwen3.5-0.8b', 'fp32')]:
    T = J(t, f'taskmod_{dts}.json')
    s.append(f"**{t} {dts}**：指令 A = `{T['instructions'][0]}`，B = `{T['instructions'][1]}`（token 长度 {T['instr_token_lens']}），attn={T['attn_implementation']}，"
             f"图像 token {T['image_first']['n_image_tokens']} 个，seq_len {T['image_first']['seq_len']}。")
    s.append('')
    s.append('| 层 | 图在前：torch.equal | 图在前：max\\|Δ\\| | 指令在前：torch.equal | 指令在前：max\\|Δ\\| | 指令在前：cos（展平） | 指令在前：逐 token cos 最小 | 指令在前：变化的 token 数 |')
    s.append('|---|---|---|---|---|---|---|---|')
    for a, b in zip(T['image_first']['layers'], T['instruction_first']['layers']):
        s.append(f"| {a['layer']} | {a['equal']} | {a['max_abs_delta']:.4g} | {b['equal']} | {b['max_abs_delta']:.4g} | {b['cos_flat']:.6f} | {b['cos_min_token']:.6f} | {b['n_tokens_changed']} |")
    s.append('')
    s.append(f"图在前：{sum(r['equal'] for r in T['image_first']['layers'])}/{len(T['image_first']['layers'])} 层逐位相同；指令在前：{sum(r['equal'] for r in T['instruction_first']['layers'])}/{len(T['instruction_first']['layers'])} 层逐位相同。")
    s.append('')
out['5_7'] = '\n'.join(s)

# ---- 5.8
s = ['| 模型 | 测量 | batch | 峰值 allocated GiB | 峰值 reserved GiB | 备注 |', '|---|---|---|---|---|---|']
for t in F:
    I = J(t, 'mem_infer.json'); L = J(t, 'mem_lora.json')
    s.append(f"| {t} | 仅加载权重（bf16） | — | {I['after_load']['alloc_GiB']} | {I['after_load']['peak_reserved_GiB']} | |")
    for bs in [1, 16]:
        for nm in ['logits_to_keep=1', 'full_logits']:
            d = I[f'forward_bs{bs}_{nm}']
            s.append(f"| {t} | 推理前向 bf16，{I['n_visual_tokens']} 视觉 token（seq {I['seq_len']}），{nm} | {bs} | {d['peak_alloc_GiB']} | {d['peak_reserved_GiB']} | logits {d['logits_shape']} |")
    for nm in ['logits_to_keep=1', 'full_logits']:
        d = I[f'probe_forward_bs8_all_hidden_{nm}']
        s.append(f"| {t} | 在线探针前向 output_hidden_states=True（{d['n_hidden_states']} 个 hidden），{nm} | 8 | {d['peak_alloc_GiB']} | {d['peak_reserved_GiB']} | 全部 hidden 合计 {d['hidden_bytes_all_layers_GiB']} GiB |")
    for tn in ['all-linear', 'language_model_linear_only']:
        d = L[f'lora_r32_{tn}_bs8_gc']
        s.append(f"| {t} | LoRA r=32（{tn}），梯度检查点，单步前向+反向，全序列 LM loss（seq {L['seq_len']}） | 8 | {d['peak_alloc_GiB']} | {d['peak_reserved_GiB']} | 可训练参数 {d['trainable_params']:,}；LoRA 模块 {d['n_lora_modules']}（其中视觉塔 {d['n_lora_modules_in_visual']}） |")
    g = I['generate_bs16_max_new_64']
    r = g['runs']
    s.append(f"| {t} | HF generate，greedy，思考关，max_new_tokens=64 | 16 | {max(x['peak_alloc_GiB'] for x in r)} | {max(x['peak_reserved_GiB'] for x in r)} | 3 次：{', '.join(str(x['samples_per_s']) for x in r)} 样本/s（墙钟 {', '.join(str(x['wall_s']) for x in r)} s）；每样本生成 {r[0]['gen_tokens_per_sample_minmax']} token；prompt {g['prompt_len']} token |")
out['5_8'] = '\n'.join(s)
json.dump(out, open(os.path.join(P, 'render.json'), 'w'), indent=1, ensure_ascii=False)
for k, v in out.items():
    if isinstance(v, str):
        open(os.path.join(P, f'render_{k}.md'), 'w').write(v)
print('ok', list(out))
