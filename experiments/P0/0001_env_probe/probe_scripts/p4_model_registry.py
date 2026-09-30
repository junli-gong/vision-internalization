"""RUN-0001 §4: model registry via HF API. No weights downloaded (only config/json/md files)."""
import json, os, sys, collections
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
api = HfApi()
OUT = os.path.join(os.environ['PROJ'], 'logs/0001/p4')
os.makedirs(OUT, exist_ok=True)

CANDS = {
  'qwen3.5': ['Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-2B', 'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-9B',
              'Qwen/Qwen3.5-0.8B-Base', 'Qwen/Qwen3.5-2B-Base', 'Qwen/Qwen3.5-4B-Base', 'Qwen/Qwen3.5-9B-Base'],
  'gemma4': ['google/gemma-4-E2B-it', 'google/gemma-4-E4B-it', 'google/gemma-4-E2B', 'google/gemma-4-E4B',
             'google/gemma-4-E2B-it-assistant', 'google/gemma-4-E4B-it-assistant'],
  'internvl3.5': [f'OpenGVLab/InternVL3_5-{s}{v}' for s in ['2B', '4B', '8B']
                  for v in ['', '-HF', '-Instruct', '-MPO', '-Pretrained', '-Flash']],
}
SMALL = ['config.json', 'generation_config.json', 'preprocessor_config.json', 'processor_config.json',
         'video_preprocessor_config.json', 'tokenizer_config.json', 'chat_template.jinja', 'chat_template.json',
         'README.md', 'LICENSE', 'special_tokens_map.json', 'added_tokens.json']

def pick(d, keys):
    return {k: d[k] for k in keys if k in d}

TXT_KEYS = ['model_type', 'num_hidden_layers', 'hidden_size', 'intermediate_size', 'num_attention_heads',
            'num_key_value_heads', 'head_dim', 'vocab_size', 'full_attention_interval', 'layer_types',
            'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim', 'linear_value_head_dim',
            'linear_conv_kernel_dim', 'sliding_window', 'max_position_embeddings', 'rope_scaling', 'rope_parameters',
            'tie_word_embeddings', 'attn_output_gate', 'hidden_size_per_layer_input', 'num_kv_shared_layers',
            'global_head_dim', 'torch_dtype', 'dtype']
VIS_KEYS = ['model_type', 'depth', 'num_hidden_layers', 'hidden_size', 'patch_size', 'spatial_merge_size',
            'temporal_patch_size', 'image_size', 'out_hidden_size', 'num_heads', 'num_attention_heads',
            'deepstack_visual_indexes', 'pooling_kernel_size', 'default_output_length', 'position_embedding_size']

def summarize_layer_types(lt):
    if not isinstance(lt, list):
        return lt
    c = collections.Counter(lt)
    return {'counts': dict(c), 'sequence': lt}

res = {}
for fam, repos in CANDS.items():
    for r in repos:
        e = {'family': fam}
        try:
            mi = api.model_info(r, files_metadata=True)
        except RepositoryNotFoundError as ex:
            e['error'] = f'not found: {ex.__class__.__name__}'; res[r] = e; print(r, e['error']); continue
        except Exception as ex:
            e['error'] = repr(ex)[:300]; res[r] = e; print(r, e['error']); continue
        e['sha'] = mi.sha
        e['last_modified'] = str(mi.last_modified)
        e['gated'] = mi.gated
        e['private'] = mi.private
        cd = mi.card_data.to_dict() if mi.card_data else {}
        e['license'] = cd.get('license'); e['license_name'] = cd.get('license_name'); e['license_link'] = cd.get('license_link')
        e['base_model'] = cd.get('base_model'); e['pipeline_tag'] = mi.pipeline_tag; e['library_name'] = mi.library_name
        e['tags'] = [t for t in (mi.tags or []) if not t.startswith('arxiv') and not t.startswith('region')][:30]
        files = {s.rfilename: s.size for s in mi.siblings}
        e['n_files'] = len(files)
        e['safetensors_bytes'] = sum(v or 0 for k, v in files.items() if k.endswith('.safetensors'))
        e['safetensors_files'] = sorted(k for k in files if k.endswith('.safetensors'))
        e['other_big_files'] = {k: v for k, v in files.items() if (v or 0) > 50e6 and not k.endswith('.safetensors')}
        e['py_files'] = sorted(k for k in files if k.endswith('.py'))
        e['safetensors_param_meta'] = None
        try:
            if mi.safetensors: e['safetensors_param_meta'] = {'total': mi.safetensors.total, 'parameters': dict(mi.safetensors.parameters)}
        except Exception: pass
        # access check: download a small file
        d = os.path.join(OUT, r.replace('/', '__')); os.makedirs(d, exist_ok=True)
        got = []
        for f in SMALL:
            if f in files:
                try:
                    p = hf_hub_download(r, f, revision=mi.sha, local_dir=d)
                    got.append(f)
                except GatedRepoError as ex:
                    e['access'] = 'gated-denied'; break
                except Exception as ex:
                    e.setdefault('dl_errors', {})[f] = repr(ex)[:200]
        e['access'] = e.get('access', 'ok' if got else 'no-small-files')
        e['downloaded'] = got
        if 'config.json' in got:
            cfg = json.load(open(os.path.join(d, 'config.json')))
            e['architectures'] = cfg.get('architectures'); e['model_type'] = cfg.get('model_type')
            tc = cfg.get('text_config') or cfg.get('llm_config') or cfg
            vc = cfg.get('vision_config') or {}
            t = pick(tc, TXT_KEYS)
            if 'layer_types' in t: t['layer_types'] = summarize_layer_types(t['layer_types'])
            e['text'] = t
            e['vision'] = pick(vc, VIS_KEYS)
            e['top_other'] = pick(cfg, ['image_token_id', 'image_token_index', 'video_token_id', 'vision_start_token_id',
                                         'vision_end_token_id', 'downsample_ratio', 'force_image_size', 'template',
                                         'ps_version', 'select_layer', 'projector_hidden_act', 'image_seq_length',
                                         'mm_tokens_per_image', 'tie_word_embeddings', 'boi_token_id', 'eoi_token_id',
                                         'image_token_index', 'vision_feature_layer', 'audio_config'])
            if 'audio_config' in e['top_other']:
                e['top_other']['audio_config'] = bool(e['top_other']['audio_config'])
        res[r] = e
        print(r, e['sha'][:10], e['license'], 'gated=', e['gated'], e['access'], f"{e['safetensors_bytes']/1e9:.2f}GB", e.get('architectures'))
json.dump(res, open(os.path.join(OUT, 'registry.json'), 'w'), indent=1, default=str)
