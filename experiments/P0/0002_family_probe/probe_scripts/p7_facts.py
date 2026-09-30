"""RUN-0002 §3.1-3.4: structure, tokenizer, chat template, image tokenization, positions for Gemma 4 / InternVL3.5-HF.
usage: python p7_facts.py gemma|internvl <repo> <revision> <tag>
Raw outputs -> $PROJ/logs/0002/<tag>/
"""
import os, sys, json, re, io, logging, warnings, itertools, inspect
import torch
import numpy as np
from PIL import Image
import transformers
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, AutoConfig, GenerationConfig

fam, repo, rev, tag = sys.argv[1:5]
OUT = os.path.join(os.environ['PROJ'], 'logs/0002', tag)
os.makedirs(OUT, exist_ok=True)
COCO = os.path.join(os.environ['PROJ'], 'tmp/coco')
R = {'family': fam, 'repo': repo, 'revision': rev, 'transformers': transformers.__version__, 'torch': torch.__version__}
log_stream = io.StringIO()
hd = logging.StreamHandler(log_stream); hd.setLevel(logging.WARNING)
transformers.logging.get_logger().addHandler(hd)
warnings.simplefilter('always')


def dump(name, obj):
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj, indent=1, default=str, ensure_ascii=False))


def rle(ids, tok):
    out = []
    for k, g in itertools.groupby(ids):
        n = len(list(g))
        t = tok.convert_ids_to_tokens(int(k))
        out.append(f'{t!r}x{n}' if n > 1 else repr(t))
    return out


with warnings.catch_warnings(record=True) as wl:
    cfg = AutoConfig.from_pretrained(repo, revision=rev)
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    model = AutoModelForImageTextToText.from_pretrained(repo, revision=rev, dtype=torch.bfloat16, device_map='cuda').eval()
R['load_warnings'] = sorted(set(str(w.message)[:300] for w in wl))
R['load_log'] = log_stream.getvalue()[-3000:]
R['classes'] = {'model': type(model).__name__, 'processor': type(proc).__name__,
                'image_processor': type(proc.image_processor).__name__, 'tokenizer': type(proc.tokenizer).__name__}
dump('model_repr.txt', repr(model))
tok = proc.tokenizer
tc = cfg.get_text_config()
vc = cfg.vision_config
lm = model.model.language_model
R['attn_implementation'] = model.config._attn_implementation

# ---------------- 3.1 structure ----------------
layers = []
for i, l in enumerate(lm.layers):
    a = l.self_attn
    d = {'idx': i, 'attn_class': type(a).__name__}
    if fam == 'gemma':
        d.update({'layer_type': tc.layer_types[i], 'is_kv_shared_layer': bool(getattr(a, 'is_kv_shared_layer', False)),
                  'store_full_length_kv': bool(getattr(a, 'store_full_length_kv', False)), 'head_dim': a.head_dim,
                  'has_k_proj': hasattr(a, 'k_proj'), 'sliding_window': getattr(a, 'sliding_window', None),
                  'mlp_intermediate': l.mlp.gate_proj.out_features})
    else:
        d.update({'layer_type': (tc.layer_types[i] if getattr(tc, 'layer_types', None) else 'full_attention'),
                  'head_dim': getattr(a, 'head_dim', None)})
    layers.append(d)
R['n_layers'] = len(layers)
R['layers'] = layers
R['layer_type_counts'] = {k: sum(1 for x in layers if x['layer_type'] == k) for k in set(x['layer_type'] for x in layers)}
R['full_attention_layer_idx'] = [x['idx'] for x in layers if x['layer_type'] == 'full_attention']
def sget(o, k):
    try:
        return getattr(o, k, None)
    except Exception as e:
        return f'<{type(e).__name__}: per-layer attribute>'
R['text_cfg'] = {k: sget(tc, k) for k in ['hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
                                                   'head_dim', 'global_head_dim', 'sliding_window', 'num_kv_shared_layers',
                                                   'hidden_size_per_layer_input', 'vocab_size', 'vocab_size_per_layer_input',
                                                   'use_bidirectional_attention', 'use_double_wide_mlp', 'final_logit_softcapping',
                                                   'rope_parameters', 'rope_theta', 'rope_scaling', 'max_position_embeddings',
                                                   'tie_word_embeddings', 'eos_token_id', 'pad_token_id', 'bos_token_id']}
if fam == 'gemma' and hasattr(tc, 'per_layer_config'):
    plc = tc.per_layer_config
    R['per_layer_config_sample'] = {}
    for key in [0, tc.layer_types.index('full_attention')]:
        try:
            c_ = plc[key]
            R['per_layer_config_sample'][f'layer{key}({tc.layer_types[key]})'] = {k: sget(c_, k) for k in ['head_dim', 'num_key_value_heads', 'sliding_window', 'rope_parameters']}
        except Exception as e:
            R['per_layer_config_sample'][f'layer{key}'] = repr(e)[:200]
if fam == 'gemma':
    R['kv_shared'] = {'first_kv_shared_layer_idx': tc.num_hidden_layers - tc.num_kv_shared_layers,
                      'kv_shared_layers': [x['idx'] for x in layers if x['is_kv_shared_layer']],
                      'kv_source_layers(store_full_length_kv)': {x['layer_type']: x['idx'] for x in layers if x['store_full_length_kv']},
                      'src_code_comment': 'modeling_gemma4.Gemma4TextAttention: "For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer."'}
    R['vision'] = {k: getattr(vc, k, None) for k in ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'patch_size',
                                                     'pooling_kernel_size', 'default_output_length', 'position_embedding_size']}
    R['vision']['px_per_llm_token_side'] = vc.patch_size * vc.pooling_kernel_size
else:
    R['vision'] = {k: getattr(vc, k, None) for k in ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'patch_size', 'image_size']}
    R['vision'].update({'downsample_ratio': getattr(cfg, 'downsample_ratio', None), 'vision_feature_layer': getattr(cfg, 'vision_feature_layer', None),
                        'projector_hidden_act': getattr(cfg, 'projector_hidden_act', None)})
    ps = vc.patch_size[0] if isinstance(vc.patch_size, (list, tuple)) else vc.patch_size
    isz = vc.image_size[0] if isinstance(vc.image_size, (list, tuple)) else vc.image_size
    R['vision']['tokens_per_tile'] = int((isz // ps) ** 2 * cfg.downsample_ratio ** 2)
    R['vision']['px_per_llm_token_side'] = int(ps / cfg.downsample_ratio)
R['param_counts'] = {'total': sum(p.numel() for p in model.parameters()),
                     'language_model': sum(p.numel() for p in lm.parameters())}
for nm in ['vision_tower', 'visual', 'audio_tower', 'multi_modal_projector', 'embed_vision', 'embed_audio']:
    if hasattr(model.model, nm):
        R['param_counts'][nm] = sum(p.numel() for p in getattr(model.model, nm).parameters())
E = lm.embed_tokens.weight.detach().float()
W = model.lm_head.weight.detach().float()
R['vocab'] = {'config_vocab_size': tc.vocab_size, 'embed_rows': E.shape[0], 'lm_head_rows': W.shape[0],
              'len_tokenizer': len(tok), 'tokenizer_vocab_size_base': tok.vocab_size,
              'lm_head_is_embed(same storage)': model.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr(),
              'embed_class': type(lm.embed_tokens).__name__, 'embed_scale': getattr(lm.embed_tokens, 'embed_scale', None)}
if hasattr(lm, 'embed_tokens_per_layer'):
    R['vocab']['embed_tokens_per_layer_shape'] = list(lm.embed_tokens_per_layer.weight.shape)
reserved = list(range(len(tok), E.shape[0]))
R['vocab']['n_rows_beyond_tokenizer'] = len(reserved)


def rowstats(M, idx):
    if not idx:
        return None
    sub = M[idx]; nrm = sub.norm(dim=1)
    return {'n': len(idx), 'norm_min': nrm.min().item(), 'norm_max': nrm.max().item(), 'norm_mean': nrm.mean().item(),
            'n_exact_zero_rows': int((sub.abs().sum(1) == 0).sum()), 'max_abs_diff_to_first_row': (sub - sub[0:1]).abs().max().item()}
g = torch.Generator().manual_seed(0)
samp = torch.randperm(min(tok.vocab_size, E.shape[0]), generator=g)[:20000].tolist()
R['embed_row_stats'] = {'base_vocab_sample20k': rowstats(E, samp), 'beyond_tokenizer_rows': rowstats(E, reserved)}
# special / unused tokens
vocab = tok.get_vocab()
unused = sorted([(t, i) for t, i in vocab.items() if re.fullmatch(r'<unused\d+>|\[unused\d+\]|<\|reserved[^>]*\|>|<reserved[^>]*>', t)], key=lambda x: x[1])
R['unused_like_tokens'] = {'n': len(unused), 'first5': unused[:5], 'last5': unused[-5:],
                           'id_range': [unused[0][1], unused[-1][1]] if unused else None}
if unused:
    ids_u = [i for _, i in unused if i < E.shape[0]]
    R['unused_like_tokens']['embed_row_stats'] = rowstats(E, ids_u)
added = tok.added_tokens_decoder
tpl = proc.chat_template if isinstance(proc.chat_template, str) else json.dumps(proc.chat_template)
R['n_added_tokens_decoder'] = len(added)
R['added_special_tokens(non-unused)'] = [{'id': i, 'content': t.content, 'special': t.special, 'in_chat_template': t.content in tpl,
                                          'embed_norm': E[i].norm().item() if i < E.shape[0] else None}
                                         for i, t in sorted(added.items()) if not re.fullmatch(r'<unused\d+>', t.content)][:400]
R['special_tokens_map'] = tok.special_tokens_map
R['named_mm_tokens'] = {k: (getattr(proc, k, None), getattr(proc, k + '_id', None)) for k in
                        ['image_token', 'boi_token', 'eoi_token', 'start_image_token', 'end_image_token', 'video_token', 'audio_token']}
# new token
t2 = AutoTokenizer.from_pretrained(repo, revision=rev)
before = len(t2)
n_add = t2.add_tokens(['<g0>']); n_add_sp = t2.add_tokens(['<g1>'], special_tokens=True)
gid0, gid1 = t2.convert_tokens_to_ids('<g0>'), t2.convert_tokens_to_ids('<g1>')
tests = {}
for s in ['<g0>', 'a<g0>b', ' <g0>', '<g0><g0><g0>', '<g0>\n<g1>', 'x <g1> y', '<g1><g0>', '12<g0>34']:
    ids = t2(s, add_special_tokens=False)['input_ids']
    tests[s] = {'ids': ids, 'tokens': t2.convert_ids_to_tokens(ids), 'decode': t2.decode(ids)}
R['new_token_test'] = {'len_before': before, 'n_added_normal': n_add, 'n_added_special': n_add_sp, 'id_<g0>': gid0, 'id_<g1>': gid1,
                       'len_after': len(t2), 'new_id_lt_embed_rows': gid1 < E.shape[0], 'tokenization': tests,
                       'embed_row_at_<g0>_norm': E[gid0].norm().item() if gid0 < E.shape[0] else None}

# ---------------- 3.2 chat template ----------------
img_path = os.path.join(COCO, '000000039769.jpg')
m_img = [{'role': 'user', 'content': [{'type': 'image', 'path': img_path}, {'type': 'text', 'text': 'Describe the image.'}]}]
m_txt = [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is 2+3?'}]}]
R1 = ("You are an AI assistant that rigorously follows this response protocol:\n\n1. First, conduct a detailed analysis of the question. "
      "Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.\n\n"
      "2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.\n\n"
      "Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.")
variants = [('default(no kwarg)', {}, None), ('enable_thinking=False', {'enable_thinking': False}, None),
            ('enable_thinking=True', {'enable_thinking': True}, None),
            ('chat_template_kwargs={enable_thinking:False}', {'chat_template_kwargs': {'enable_thinking': False}}, None)]
if fam == 'internvl':
    variants.append(('system=R1_SYSTEM_PROMPT (card thinking mode)', {}, R1))
renders = {}
for name, kw, sysmsg in variants:
    for mname, m in [('text', m_txt), ('image', m_img)]:
        msgs = ([{'role': 'system', 'content': [{'type': 'text', 'text': sysmsg}]}] if sysmsg else []) + m
        log_stream.truncate(0); log_stream.seek(0)
        try:
            with warnings.catch_warnings(record=True) as w2:
                s = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            renders[f'{name} | {mname}'] = {'text': s, 'warnings': sorted(set(str(x.message)[:200] for x in w2)), 'log': log_stream.getvalue()[-400:]}
        except Exception as e:
            renders[f'{name} | {mname}'] = {'error': repr(e)[:400]}
R['template_variables'] = sorted(transformers.utils.chat_template_utils._get_template_variables(tpl)) if hasattr(transformers.utils, 'chat_template_utils') else None
R['chat_template_renders'] = renders
R['generation_config'] = {k: v for k, v in model.generation_config.to_dict().items() if v is not None}
try:
    GenerationConfig.from_pretrained(repo, revision=rev); R['generation_config_has_file'] = True
except Exception as e:
    R['generation_config_has_file'] = False
eos = model.generation_config.eos_token_id
eos = eos if isinstance(eos, list) else [eos]
R['eos_tokens'] = {int(i): tok.convert_ids_to_tokens(int(i)) for i in eos if i is not None}
R['config_eos'] = {'top': getattr(cfg, 'eos_token_id', None), 'text': getattr(tc, 'eos_token_id', None)}

# ---------------- 3.3 image tokenization ----------------
ip = proc.image_processor
R['image_processor_attrs'] = {k: getattr(ip, k, None) for k in ['size', 'max_soft_tokens', 'patch_size', 'pooling_kernel_size', 'image_seq_length',
                                                                  'crop_to_patches', 'min_patches', 'max_patches', 'do_resize', 'resample']}
R['processor_default_images_kwargs'] = None
try:
    kw_cls = [c for n, c in inspect.getmembers(sys.modules[type(proc).__module__]) if n.endswith('ProcessorKwargs')]
    R['processor_default_images_kwargs'] = {c.__name__: getattr(c, '_defaults', None) for c in kw_cls}
except Exception as e:
    R['processor_default_images_kwargs'] = repr(e)
img_tok_id = proc.image_token_id
rng = np.random.default_rng(0)
synth = lambda w, h: Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def one(img, **kw):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'hi'}]}]
    extra = {'enable_thinking': False} if fam == 'gemma' else {}
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra)
    enc = proc(text=[text], images=[img], return_tensors='pt', **kw)
    n = int((enc['input_ids'][0] == img_tok_id).sum())
    d = {'input_wh': list(img.size), 'n_image_tokens': n, 'pixel_values_shape': list(enc['pixel_values'].shape), 'keys': list(enc.keys())}
    if fam == 'gemma':
        pos = enc['image_position_ids'][0]
        valid = pos[(pos >= 0).all(-1)]
        hp, wp = int(valid[:, 1].max()) + 1, int(valid[:, 0].max()) + 1  # (x, y) order to be verified below
        d.update({'num_soft_tokens_per_image': enc.get('num_soft_tokens_per_image', None) if not torch.is_tensor(enc.get('num_soft_tokens_per_image', None)) else enc['num_soft_tokens_per_image'].tolist(),
                  'n_valid_patches': int(valid.shape[0]), 'n_padded_patches': int(pos.shape[0] - valid.shape[0]),
                  'patch_grid_from_position_ids(dim1+1, dim0+1)': [hp, wp],
                  'position_ids_first3': pos[:3].tolist(), 'position_ids_valid_last2': valid[-2:].tolist()})
        # resized image size in px = patch grid * 16; LLM grid = patch grid / 3
        d['resized_px_(a,b)'] = [hp * 16, wp * 16]
        d['llm_grid_(a,b)'] = [hp // 3, wp // 3]
    else:
        d['n_tiles'] = int(enc['pixel_values'].shape[0])
    return d


tk = {}
sizes = [(448, 448), (640, 480), (1024, 768), (1024, 1024)]
if fam == 'gemma':
    for (w, h) in sizes:
        tk[f'{w}x{h} default'] = one(synth(w, h))
        for b in [70, 140, 280, 560, 1120]:
            try:
                tk[f'{w}x{h} max_soft_tokens={b}'] = one(synth(w, h), max_soft_tokens=b)
            except Exception as e:
                tk[f'{w}x{h} max_soft_tokens={b}'] = {'error': repr(e)[:300]}
    try:
        tk['640x480 images_kwargs={max_soft_tokens:70}'] = one(synth(640, 480), images_kwargs={'max_soft_tokens': 70})
    except Exception as e:
        tk['640x480 images_kwargs={max_soft_tokens:70}'] = {'error': repr(e)[:300]}
    tk['500x333 default'] = one(synth(500, 333))
    tk['2000x300 default'] = one(synth(2000, 300))
else:
    for (w, h) in sizes + [(500, 333), (2000, 300)]:
        tk[f'{w}x{h} default'] = one(synth(w, h))
        for ctp in [False, True]:
            for mp in ([12] if not ctp else [1, 4, 6, 12]):
                try:
                    tk[f'{w}x{h} crop_to_patches={ctp} max_patches={mp}'] = one(synth(w, h), crop_to_patches=ctp, max_patches=mp)
                except Exception as e:
                    tk[f'{w}x{h} crop_to_patches={ctp} max_patches={mp}'] = {'error': repr(e)[:300]}
R['image_tokenization'] = tk
# two images
msgs2 = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text': 'Compare.'}]}]
extra = {'enable_thinking': False} if fam == 'gemma' else {}
text2 = proc.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True, **extra)
enc2 = proc(text=[text2], images=[synth(448, 448), synth(640, 480)], return_tensors='pt')
R['two_images'] = {'template_text': text2, 'token_rle': rle(enc2['input_ids'][0].tolist(), tok),
                   'mm_token_type_ids_rle': [f'{k}x{len(list(g_))}' for k, g_ in itertools.groupby(enc2['mm_token_type_ids'][0].tolist())] if 'mm_token_type_ids' in enc2 else None,
                   'keys': list(enc2.keys()), 'pixel_values_shape': list(enc2['pixel_values'].shape)}

# ---------------- 3.4 positions ----------------
cap = {}
def pre_hook(mod, args, kwargs):
    if 'position_ids' not in cap:
        pid = kwargs.get('position_ids', None)
        if pid is None and len(args) >= 2:
            pid = args[1]
        cap['position_ids'] = None if pid is None else pid.detach().cpu().clone()
    return None
hh = lm.rotary_emb.register_forward_pre_hook(pre_hook, with_kwargs=True)
msgs3 = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Find the cat.'}, {'type': 'image'}, {'type': 'text', 'text': 'Answer briefly.'}]}]
text3 = proc.apply_chat_template(msgs3, tokenize=False, add_generation_prompt=True, **extra)
enc3 = proc(text=[text3], images=[synth(448, 320)], return_tensors='pt').to('cuda')
with torch.no_grad():
    out3 = model(**enc3, output_hidden_states=True)
hh.remove()
ids3 = enc3['input_ids'][0].tolist()
pid = cap.get('position_ids')
isimg = [int(t == img_tok_id) for t in ids3]
segs = []
for k, grp in itertools.groupby(range(len(ids3)), key=lambda j: isimg[j]):
    idx = list(grp)
    p = pid[..., idx] if pid is not None else None
    segs.append({'type': 'image' if k else 'text', 'tok_range': [idx[0], idx[-1]], 'n': len(idx),
                 'first_tokens': tok.convert_ids_to_tokens(ids3[idx[0]:idx[0] + 3]),
                 'pos_first5': p.reshape(-1, len(idx))[:, :5].tolist() if p is not None else None,
                 'pos_last3': p.reshape(-1, len(idx))[:, -3:].tolist() if p is not None else None})
R['positions'] = {'example_text': text3, 'seq_len': len(ids3), 'position_ids_shape_at_rotary': list(pid.shape) if pid is not None else None,
                  'position_ids_is_arange': bool(torch.equal(pid.reshape(-1), torch.arange(len(ids3)))) if pid is not None else None,
                  'segments': segs}
if fam == 'gemma':
    ipos = enc3['image_position_ids'][0].cpu()
    valid = ipos[(ipos >= 0).all(-1)]
    R['positions']['vision_encoder_image_position_ids'] = {'shape': list(enc3['image_position_ids'].shape), 'first6': valid[:6].tolist(),
                                                           'last3': valid[-3:].tolist(), 'dim0_max': int(valid[:, 0].max()), 'dim1_max': int(valid[:, 1].max()),
                                                           'n_valid': int(valid.shape[0]), 'n_pad(-1)': int(ipos.shape[0] - valid.shape[0])}
# output_hidden_states structure
hs = out3.hidden_states
R['output_hidden_states'] = {'n': len(hs), 'shapes': [list(h.shape) for h in hs][:3] + ['...'] + [list(h.shape) for h in hs][-2:],
                             'all_same_shape': len(set(tuple(h.shape) for h in hs)) == 1,
                             'n_layers+1': R['n_layers'] + 1}
if fam == 'gemma':
    with torch.no_grad():
        o_lm = lm(input_ids=None, inputs_embeds=None) if False else None
    R['output_hidden_states']['note'] = 'hidden_states recorded by capture_outputs at Gemma4TextDecoderLayer outputs; per_layer_inputs (PLE) are an extra input to each layer, not a separate returned tensor'

# resize_token_embeddings behaviour (mutates model; last)
try:
    rows0 = model.get_input_embeddings().weight.shape[0]
    ple0 = list(lm.embed_tokens_per_layer.weight.shape) if hasattr(lm, 'embed_tokens_per_layer') else None
    model.resize_token_embeddings(len(t2))
    R['resize_token_embeddings(len(tok_with_2_new))'] = {'rows_before': rows0, 'rows_after': model.get_input_embeddings().weight.shape[0],
                                                          'lm_head_rows_after': model.lm_head.weight.shape[0],
                                                          'ple_before': ple0, 'ple_after': list(lm.embed_tokens_per_layer.weight.shape) if hasattr(lm, 'embed_tokens_per_layer') else None,
                                                          'config_vocab_after': model.config.get_text_config().vocab_size}
except Exception as e:
    R['resize_token_embeddings(len(tok_with_2_new))'] = {'error': repr(e)[:400]}
R['captured_log'] = log_stream.getvalue()[-3000:]
dump('facts.json', R)
print(json.dumps({k: R[k] for k in ['classes', 'n_layers', 'layer_type_counts', 'full_attention_layer_idx', 'text_cfg', 'vision', 'vocab',
                                    'embed_row_stats', 'unused_like_tokens', 'param_counts', 'output_hidden_states', 'eos_tokens',
                                    'resize_token_embeddings(len(tok_with_2_new))'] + (['kv_shared'] if fam == 'gemma' else [])}, indent=1, default=str))
