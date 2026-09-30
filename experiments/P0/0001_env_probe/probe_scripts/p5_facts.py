"""RUN-0001 §5.1-5.4: Qwen3.5 structure, tokenizer, chat template, image tokenization, M-RoPE positions.
usage: python p5_facts.py <repo_id> <tag>
Raw outputs -> $PROJ/logs/0001/p5/<tag>/
"""
import os, sys, json, re, io, contextlib, logging, warnings, itertools
import torch
import numpy as np
from PIL import Image
import transformers
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoConfig

repo, tag = sys.argv[1], sys.argv[2]
OUT = os.path.join(os.environ['PROJ'], 'logs/0001/p5', tag)
os.makedirs(OUT, exist_ok=True)
COCO = os.path.join(os.environ['PROJ'], 'tmp/coco')
R = {'repo': repo, 'transformers': transformers.__version__, 'torch': torch.__version__}

# capture transformers warnings
log_stream = io.StringIO()
h = logging.StreamHandler(log_stream); h.setLevel(logging.WARNING)
transformers.logging.get_logger().addHandler(h)
warnings.simplefilter('always')


def dump(name, obj):
    with open(os.path.join(OUT, name), 'w') as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, indent=1, default=str, ensure_ascii=False)


def rle(ids, tok):
    out = []
    for k, g in itertools.groupby(ids):
        n = len(list(g))
        out.append(f"{tok.convert_ids_to_tokens(int(k))!r}x{n}" if n > 1 else repr(tok.convert_ids_to_tokens(int(k))))
    return out


with warnings.catch_warnings(record=True) as wlist:
    cfg = AutoConfig.from_pretrained(repo)
    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map='cuda').eval()
R['load_warnings'] = [str(w.message)[:300] for w in wlist]
R['model_class'] = type(model).__name__
R['processor_class'] = type(proc).__name__
R['image_processor_class'] = type(proc.image_processor).__name__
R['tokenizer_class'] = type(proc.tokenizer).__name__
dump('model_repr.txt', repr(model))
tok = proc.tokenizer

# ---------------- 5.1 structure ----------------
tc = cfg.text_config
lm = model.model.language_model
layers = []
for i, l in enumerate(lm.layers):
    mixer = l.linear_attn if hasattr(l, 'linear_attn') else l.self_attn
    layers.append({'idx': i, 'block_type': l.block_type, 'mixer_class': type(mixer).__name__})
R['n_layers'] = len(lm.layers)
R['layer_types_seq'] = [x['block_type'] for x in layers]
R['layer_type_counts'] = {k: R['layer_types_seq'].count(k) for k in set(R['layer_types_seq'])}
R['full_attention_layer_idx'] = [x['idx'] for x in layers if x['block_type'] == 'full_attention']
R['text_hidden_size'] = tc.hidden_size
R['full_attn'] = {'num_attention_heads': tc.num_attention_heads, 'num_key_value_heads': tc.num_key_value_heads,
                  'head_dim': tc.head_dim, 'attn_output_gate': getattr(tc, 'attn_output_gate', None)}
R['gdn'] = {k: getattr(tc, k, None) for k in ['linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim',
                                               'linear_value_head_dim', 'linear_conv_kernel_dim']}
vc = cfg.vision_config
R['vision'] = {k: getattr(vc, k, None) for k in ['depth', 'hidden_size', 'num_heads', 'patch_size', 'spatial_merge_size',
                                                 'temporal_patch_size', 'out_hidden_size', 'deepstack_visual_indexes']}
R['vision']['pixels_per_llm_token_side'] = vc.patch_size * vc.spatial_merge_size
R['param_counts'] = {
    'total': sum(p.numel() for p in model.parameters()),
    'visual': sum(p.numel() for p in model.model.visual.parameters()),
    'language_model': sum(p.numel() for p in lm.parameters()),
    'lm_head_separate': (model.lm_head.weight.data_ptr() != lm.embed_tokens.weight.data_ptr()),
}
# vocab / embedding rows
E = lm.embed_tokens.weight.detach().float()
W = model.lm_head.weight.detach().float()
R['vocab'] = {
    'config_vocab_size': tc.vocab_size,
    'embed_rows': E.shape[0], 'lm_head_rows': W.shape[0],
    'len_tokenizer': len(tok), 'tokenizer_vocab_size_base': tok.vocab_size,
    'tie_word_embeddings(text_config)': getattr(tc, 'tie_word_embeddings', None),
    'tie_word_embeddings(top)': getattr(cfg, 'tie_word_embeddings', None),
    'lm_head_is_embed (same storage)': model.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr(),
}
n_tok = len(tok)
reserved = list(range(n_tok, E.shape[0]))
R['vocab']['n_rows_beyond_tokenizer'] = len(reserved)
def rowstats(M, idx):
    if not idx:
        return None
    sub = M[idx]
    nrm = sub.norm(dim=1)
    return {'n': len(idx), 'norm_min': nrm.min().item(), 'norm_max': nrm.max().item(), 'norm_mean': nrm.mean().item(),
            'n_exact_zero_rows': int((sub.abs().sum(1) == 0).sum()),
            'max_abs_diff_to_first_row': (sub - sub[0:1]).abs().max().item()}
base_ids = list(range(0, tok.vocab_size))
g = torch.Generator().manual_seed(0)
samp = torch.randperm(len(base_ids), generator=g)[:20000].tolist()
R['embed_row_stats'] = {'base_vocab_sample20k': rowstats(E, samp), 'beyond_tokenizer_rows': rowstats(E, reserved)}
if R['param_counts']['lm_head_separate']:
    R['lm_head_row_stats'] = {'base_vocab_sample20k': rowstats(W, samp), 'beyond_tokenizer_rows': rowstats(W, reserved)}
# special / added tokens
added = tok.added_tokens_decoder
template_src = proc.chat_template if isinstance(proc.chat_template, str) else json.dumps(proc.chat_template)
spec = []
for i, t in sorted(added.items()):
    spec.append({'id': i, 'content': t.content, 'special': t.special,
                 'in_chat_template': t.content in template_src,
                 'embed_norm': E[i].norm().item(),
                 'lm_head_norm': W[i].norm().item()})
R['added_tokens'] = spec
R['special_tokens_map'] = tok.special_tokens_map
# special-looking strings inside the base vocab
v = tok.get_vocab()
pat = re.compile(r'^<\|[^|]+\|>$|^<[a-z_]+>$|^</[a-z_]+>$|reserved|unused', re.I)
R['special_looking_in_base_vocab'] = sorted([(k, i) for k, i in v.items() if i < tok.vocab_size and pat.search(k)], key=lambda x: x[1])[:200]

# new token test (fresh tokenizer copy, does not affect proc)
t2 = AutoTokenizer.from_pretrained(repo)
before = len(t2)
n_add = t2.add_tokens(['<g0>'])
n_add_sp = t2.add_tokens(['<g1>'], special_tokens=True)
gid0, gid1 = t2.convert_tokens_to_ids('<g0>'), t2.convert_tokens_to_ids('<g1>')
tests = {}
for s in ['<g0>', 'a<g0>b', ' <g0>', '<g0><g0><g0>', '<g0>\n<g1>', 'x <g1> y', '<g1><g0>', '12<g0>34']:
    ids = t2(s, add_special_tokens=False)['input_ids']
    tests[s] = {'ids': ids, 'tokens': t2.convert_ids_to_tokens(ids), 'decode': t2.decode(ids)}
R['new_token_test'] = {'len_before': before, 'n_added_normal': n_add, 'n_added_special': n_add_sp,
                       'id_<g0>': gid0, 'id_<g1>': gid1, 'len_after': len(t2),
                       'new_id_lt_embed_rows': gid1 < E.shape[0], 'tokenization': tests,
                       'embed_row_at_<g0>_norm': E[gid0].norm().item() if gid0 < E.shape[0] else None}

# ---------------- 5.2 chat template / thinking ----------------
img_path = os.path.join(COCO, '000000039769.jpg')
msgs_txt = [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is 2+3?'}]}]
msgs_img = [{'role': 'user', 'content': [{'type': 'image', 'path': img_path}, {'type': 'text', 'text': 'Describe the image.'}]}]
renders = {}
for name, kw in [('default(no kwarg)', {}), ('enable_thinking=False', {'enable_thinking': False}),
                 ('enable_thinking=True', {'enable_thinking': True}),
                 ('chat_template_kwargs={enable_thinking:False}', {'chat_template_kwargs': {'enable_thinking': False}})]:
    for mname, m in [('text', msgs_txt), ('image', msgs_img)]:
        log_stream.truncate(0); log_stream.seek(0)
        try:
            with warnings.catch_warnings(record=True) as wl:
                s = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
            renders[f'processor | {name} | {mname}'] = {'text': s, 'warnings': [str(w.message)[:200] for w in wl],
                                                        'log': log_stream.getvalue()[-500:]}
        except Exception as e:
            renders[f'processor | {name} | {mname}'] = {'error': repr(e)[:500]}
    try:
        s = tok.apply_chat_template([{'role': 'user', 'content': 'What is 2+3?'}], tokenize=False, add_generation_prompt=True, **kw)
        renders[f'tokenizer | {name} | text(str content)'] = {'text': s}
    except Exception as e:
        renders[f'tokenizer | {name} | text(str content)'] = {'error': repr(e)[:500]}
R['chat_template_renders'] = renders
R['generation_config'] = model.generation_config.to_dict()
R['generation_config_has_file'] = False
try:
    from transformers import GenerationConfig
    GenerationConfig.from_pretrained(repo)
    R['generation_config_has_file'] = True
except Exception as e:
    R['generation_config_from_pretrained_error'] = repr(e)[:200]

# ---------------- 5.3 image tokenization ----------------
ip = proc.image_processor
R['image_processor_defaults'] = {k: getattr(ip, k, None) for k in ['size', 'min_pixels', 'max_pixels', 'patch_size', 'merge_size',
                                                                   'temporal_patch_size', 'do_resize', 'resample']}
rng = np.random.default_rng(0)
def synth(w, h):
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
img_tok = proc.image_token if hasattr(proc, 'image_token') else '<|image_pad|>'
img_tok_id = tok.convert_tokens_to_ids('<|image_pad|>')
def count_for(img, **kw):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'hi'}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = proc(text=[text], images=[img], return_tensors='pt', **kw)
    thw = enc['image_grid_thw'][0].tolist()
    n = int((enc['input_ids'][0] == img_tok_id).sum())
    m = vc.spatial_merge_size
    return {'input_wh': img.size, 'image_grid_thw(patches)': thw, 'llm_grid_hw': [thw[1] // m, thw[2] // m],
            'n_image_pad_tokens': n, 'resized_hw_px': [thw[1] * vc.patch_size, thw[2] * vc.patch_size],
            'pixel_values_shape': list(enc['pixel_values'].shape),
            'px_per_token_in_resized': (thw[1] * vc.patch_size * thw[2] * vc.patch_size) / n,
            'px_per_token_in_original': (img.size[0] * img.size[1]) / n}
tk = {}
for (w, h) in [(448, 448), (640, 480), (1024, 768), (1024, 1024)]:
    tk[f'{w}x{h} default'] = count_for(synth(w, h))
# fixing resolution: size={'shortest_edge': A, 'longest_edge': A} (these are pixel-count bounds)
for (w, h) in [(640, 480), (1024, 768)]:
    for A in [448 * 448, 1024 * 1024]:
        try:
            tk[f'{w}x{h} size=(min=max={A})'] = count_for(synth(w, h), size={'shortest_edge': A, 'longest_edge': A})
        except Exception as e:
            tk[f'{w}x{h} size=(min=max={A})'] = {'error': repr(e)[:300]}
        try:
            tk[f'{w}x{h} min_pixels=max_pixels={A}'] = count_for(synth(w, h), min_pixels=A, max_pixels=A)
        except Exception as e:
            tk[f'{w}x{h} min_pixels=max_pixels={A}'] = {'error': repr(e)[:300]}
# odd size not divisible by 32
tk['500x333 default'] = count_for(synth(500, 333))
R['image_tokenization'] = tk
# two images
msgs2 = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text': 'Compare.'}]}]
text2 = proc.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc2 = proc(text=[text2], images=[synth(448, 448), synth(640, 480)], return_tensors='pt')
R['two_images'] = {'template_text': text2, 'image_grid_thw': enc2['image_grid_thw'].tolist(),
                   'token_rle': rle(enc2['input_ids'][0].tolist(), tok),
                   'mm_token_type_ids_rle': [f'{k}x{len(list(g))}' for k, g in itertools.groupby(enc2['mm_token_type_ids'][0].tolist())] if 'mm_token_type_ids' in enc2 else None,
                   'processor_output_keys': list(enc2.keys())}
# add_vision_id variant
text2b = proc.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True, enable_thinking=False, add_vision_id=True)
R['two_images']['template_text_add_vision_id'] = text2b

# ---------------- 5.4 positions ----------------
msgs3 = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Find the cat.'}, {'type': 'image'}, {'type': 'text', 'text': 'Answer briefly.'}]}]
text3 = proc.apply_chat_template(msgs3, tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc3 = proc(text=[text3], images=[synth(448, 320)], return_tensors='pt').to('cuda')
pos, deltas = model.model.get_rope_index(enc3['input_ids'], mm_token_type_ids=enc3['mm_token_type_ids'],
                                         image_grid_thw=enc3['image_grid_thw'], attention_mask=enc3['attention_mask'])
ids3 = enc3['input_ids'][0].tolist()
types3 = enc3['mm_token_type_ids'][0].tolist()
segs = []
i = 0
for k, g in itertools.groupby(range(len(ids3)), key=lambda j: types3[j]):
    idx = list(g)
    p = pos[:, 0, idx]
    segs.append({'type': {0: 'text', 1: 'image', 2: 'video'}[k], 'tok_range': [idx[0], idx[-1]], 'n': len(idx),
                 'first_tokens': tok.convert_ids_to_tokens(ids3[idx[0]:idx[0] + 3]),
                 'last_tokens': tok.convert_ids_to_tokens(ids3[max(idx[0], idx[-1] - 2):idx[-1] + 1]),
                 'T_range': [p[0].min().item(), p[0].max().item()], 'H_range': [p[1].min().item(), p[1].max().item()],
                 'W_range': [p[2].min().item(), p[2].max().item()],
                 'first5_THW': p[:, :5].T.tolist(), 'last3_THW': p[:, -3:].T.tolist()})
R['positions'] = {'example_text': text3, 'image_grid_thw': enc3['image_grid_thw'].tolist(),
                  'get_rope_index_shape': list(pos.shape), 'rope_deltas': deltas.tolist(), 'seq_len': len(ids3),
                  'segments': segs}
# what generate() actually builds: [4, B, L] (row0 = text positions)
try:
    mk = {k: v for k, v in enc3.items() if k != 'input_ids'}
    mk['input_ids'] = enc3['input_ids']
    gp = model._prepare_position_ids_for_generation(enc3['input_ids'], dict(mk))
    R['positions']['generate_position_ids_shape'] = list(gp.shape)
    R['positions']['generate_row0_is_arange'] = bool(torch.equal(gp[0, 0].cpu(), torch.arange(gp.shape[-1])))
    R['positions']['generate_rows1to3_equal_get_rope_index'] = bool(torch.equal(gp[1:].cpu(), pos.cpu()))
except Exception as e:
    R['positions']['generate_position_ids_error'] = repr(e)[:300]

# resize_token_embeddings behaviour (done last; mutates model)
try:
    rows0 = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(t2))
    R['resize_token_embeddings(len(tok_with_2_new))'] = {'rows_before': rows0, 'rows_after': model.get_input_embeddings().weight.shape[0],
                                                          'lm_head_rows_after': model.lm_head.weight.shape[0]}
except Exception as e:
    R['resize_token_embeddings(len(tok_with_2_new))'] = {'error': repr(e)[:300]}

R['captured_log'] = log_stream.getvalue()[-3000:]
dump('facts.json', R)
print(json.dumps({k: R[k] for k in ['model_class', 'n_layers', 'layer_type_counts', 'full_attention_layer_idx', 'text_hidden_size',
                                    'full_attn', 'gdn', 'vision', 'vocab', 'embed_row_stats', 'new_token_test']}, indent=1, default=str))
