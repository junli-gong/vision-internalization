"""RUN-0004: render the requested key values of the four smoke JSONs into markdown (numbers read from JSON only).
usage: python p8_smoke_summary.py <dir with smoke_*.json> > out.md
"""
import json, sys, os
D = sys.argv[1]
FILES = ['smoke_qwen3.5-0.8b.json', 'smoke_gemma-4-e2b-it.json', 'smoke_internvl3.5-2b-hf.json', 'smoke_qwen3.5-4b.json']
esc = lambda x: str(x).replace('|', '\\|')
Rs = {f: json.load(open(os.path.join(D, f))) for f in FILES}
out = []
# ok table
names = sorted({k for R in Rs.values() for k in R['ok']})
out += ['### 3.1 各检查项的 ok（原样）', '', '| 检查 | ' + ' | '.join(R['checks']['0_facts']['model_key'] for R in Rs.values()) + ' |', '|---|' + '---|' * len(Rs)]
for n in names:
    out.append(f'| {n} | ' + ' | '.join(str(R['ok'].get(n, '—')) for R in Rs.values()) + ' |')
out.append('| **all_ok** | ' + ' | '.join(f"**{R['all_ok']}**" for R in Rs.values()) + ' |')
out.append('| wall_time_s | ' + ' | '.join(str(R['wall_time_s']) for R in Rs.values()) + ' |')
out.append('')
# check 0
out += ['### 3.2 检查 0：实际内核实现与 lm_head 共享', '']
for f, R in Rs.items():
    c = R['checks']['0_facts']
    key = c['model_key']
    tied = c.get('tied_word_embeddings', c.get('tied_word_embeddings_runtime'))
    line = f"- **{key}**（{c['repo']} @ {c['revision'][:12]}，{c['device']}，{c['dtype']}，attn={c['attn_implementation']}）：lm_head 与嵌入共享 = **{tied}**"
    if 'config_tie_word_embeddings' in c:
        line += f"（config 的 tie_word_embeddings = {json.dumps(c['config_tie_word_embeddings'])}）"
    line += f"；嵌入 {c['embed_rows']} 行 / lm_head {c['lm_head_rows']} 行 / len(tokenizer) {c['len_tokenizer']}"
    out.append(line)
    if 'kernels' in c:
        out.append(f"  - kernels（模块级绑定）：`{json.dumps(c['kernels'])}`")
    if 'gdn_in_adapter_forward' in c:
        g = c['gdn_in_adapter_forward']
        out.append('  - 适配器前向里实际用到的 GDN：' + '；'.join(f"`{k}` → `{v['module']}.{v['name']}`（is_torch_reference={v['is_torch_reference']}, switched_to_torch={v['switched_to_torch']}）" for k, v in g.items()))
    extra = {k: c[k] for k in ['max_soft_tokens', 'image_grid_hw', 'n_image_tokens', 'kv_shared_layers', 'sliding_window', 'n_unused_pool', 'n_base_rows_for_mean',
                                'tile_policy', 'tile_grid_hw', 'tokens_per_tile', 'image_size_arg_ignored', 'llm_grid_hw', 'stop_token_ids', 'r1_system_prompt_source'] if k in c}
    out.append(f"  - 其他：`{esc(json.dumps(extra, ensure_ascii=False))}`")
out.append('')
# check 1
out += ['### 3.3 检查 1：与官方 processor 的逐项比对', '', '| 模型 | 顺序 | 思考 | template_equal | 官方 processor 逐项 | 视觉 token | prompt token | thinking_unclosed | text（前 40 字） |', '|---|---|---|---|---|---|---|---|---|']
for f, R in Rs.items():
    c1 = R['checks']['1_build_generate']
    for cs in c1['cases']:
        op = cs.get('official_processor', cs.get('processor_compare', {}))
        opd = ', '.join(f'{k}={v}' for k, v in op.items())
        out.append(f"| {R['checks']['0_facts']['model_key']} | {cs['order']} | {cs['enable_thinking']} | {cs.get('template_equal')} | {esc(opd)} | {cs.get('n_visual_tokens')} | {cs.get('n_prompt_tokens')} | {cs.get('thinking_unclosed')} | {esc(repr(cs.get('text', ''))[:44])} |")
out.append('')
# check 2
out += ['### 3.4 检查 2：任务调制仪器（经适配器 API）', '']
for f, R in Rs.items():
    c2 = R['checks']['2_task_modulation']
    key = R['checks']['0_facts']['model_key']
    for order in ['image_first', 'instruction_first']:
        o = c2[order]
        mx = ' '.join(f"L{l['layer']}:{l['max_abs_delta']:.4g}" for l in o['layers'])
        out.append(f"- **{key} {order}**：逐位相同 {o['equal_layers']}（图像 token {o['n_image_tokens']}，seq {o['seq_len']}，指令 token 长度 {o['instr_token_lens']}）。逐层 max|Δ|：{mx}")
out.append('')
# check 3
out += ['### 3.5 检查 3：grid token', '', '| 模型 | ids | 单个 token | 行数 前→后 | 均值初始化（嵌入 / lm_head） | 生成全在允许集合内 | 生成的 ids |', '|---|---|---|---|---|---|---|']
for f, R in Rs.items():
    c = R['checks']['3_grid_tokens']
    mi = f"{c.get('mean_init_embed')} / {c.get('mean_init_lm_head')}"
    extra = {k: c[k] for k in c if k not in ('ok', 'ids', 'single_token', 'rows_before', 'rows_after', 'mean_init_embed', 'mean_init_lm_head', 'all_generated_in_allowed', 'generated_ids', 'text', 'wall_time_s')}
    out.append(f"| {R['checks']['0_facts']['model_key']} | {c['ids']} | {c['single_token']} | {c.get('rows_before')}→{c.get('rows_after')} | {mi} | {c['all_generated_in_allowed']} | {c['generated_ids']} |")
    if extra:
        out.append(f"| ↳ 其他字段 | `{esc(json.dumps(extra, ensure_ascii=False))}` ||||||")
out.append('')
# check 7
out += ['### 3.6 检查 7：思考段', '']
for f, R in Rs.items():
    c = R['checks']['7_thinking']
    key = R['checks']['0_facts']['model_key']
    fields = {k: c[k] for k in ['case', 'channel_opened', 'closed', 'closed_case_exercised', 'thinking_unclosed', 'system_turn_in_prompt', 'n_generated_tokens', 'max_new_tokens', 'format_facts'] if k in c}
    out.append(f"- **{key}**：`{esc(json.dumps(fields, ensure_ascii=False))}`；text=`{esc(repr(c['text'])[:120])}`；raw_text 开头=`{esc(repr(c['raw_text'])[:160])}`；关思考对照 `{esc(json.dumps(c.get('thinking_off_control'), ensure_ascii=False))}`")
out.append('')
if '9_multi_tile' in Rs['smoke_internvl3.5-2b-hf.json']['checks']:
    c9 = Rs['smoke_internvl3.5-2b-hf.json']['checks']['9_multi_tile']
    out += ['### 3.7 检查 9（仅 InternVL）：多 tile', '', f"`{esc(json.dumps({k: v for k, v in c9.items() if k != 'adapter_path_vs_model_path'}, ensure_ascii=False))}`；adapter_path_vs_model_path：{sum(c9['adapter_path_vs_model_path'])}/{len(c9['adapter_path_vs_model_path'])} 为 True", '']
print('\n'.join(out))
