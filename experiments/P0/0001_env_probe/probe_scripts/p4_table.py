"""Render §4 registry.json into markdown tables (numbers from the JSON only)."""
import json, sys, os
R = json.load(open(os.path.join(os.environ['PROJ'], 'logs/0001/p4/registry.json')))
def attn(e):
    t = e.get('text', {}); lt = t.get('layer_types')
    if isinstance(lt, dict):
        c = lt['counts']; s = ' + '.join(f'{v} {k}' for k, v in c.items())
    else:
        s = 'all full_attention (no layer_types in config)'
    heads = f"{t.get('num_attention_heads')}Q/{t.get('num_key_value_heads')}KV×{t.get('head_dim')}"
    if 'linear_num_value_heads' in t:
        heads += f"; GDN {t['linear_num_key_heads']}K/{t['linear_num_value_heads']}V heads ×{t['linear_key_head_dim']}"
    if t.get('sliding_window'):
        heads += f"; sliding_window={t['sliding_window']}"
    return s, heads
print('| repo | revision (sha) | license | gated | our token | safetensors GB | architectures | LLM layers | hidden | attention layout | heads | vision (patch, merge/pool) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for r, e in R.items():
    if 'error' in e:
        print(f'| {r} | ERROR {e["error"]} |||||||||||'); continue
    s, h = attn(e); t = e.get('text', {}); v = e.get('vision', {})
    vis = f"patch {v.get('patch_size')}, " + (f"merge {v.get('spatial_merge_size')}" if 'spatial_merge_size' in v else (f"pool {v.get('pooling_kernel_size')}, default {v.get('default_output_length')} tok" if 'pooling_kernel_size' in v else f"img {v.get('image_size')}, downsample {e.get('top_other',{}).get('downsample_ratio')}")) if v else '—'
    print(f"| `{r}` | `{e['sha']}` | {e['license']} | {e['gated']} | {e['access']} | {e['safetensors_bytes']/1e9:.2f} | {','.join(e.get('architectures') or [])} | {t.get('num_hidden_layers')} | {t.get('hidden_size')} | {s} | {h} | {vis} |")
