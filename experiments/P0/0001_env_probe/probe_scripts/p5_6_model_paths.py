"""RUN-0001 §5.6.2-3: Qwen3.5-0.8B whole-model logits, fla path vs torch fallback path; which path transformers takes.
usage: python p5_6_model_paths.py inproc     # default import (fla available), then monkeypatch to torch reference in-process
       python p5_6_model_paths.py blockfla   # fla import blocked before transformers import -> transformers' own fallback
"""
import sys, os
mode = sys.argv[1]
if mode == 'blockfla':
    sys.modules['fla'] = None  # importlib.import_module('fla') -> ImportError -> torch fallback
import io, json, hashlib, logging, warnings
import torch
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText
import transformers.models.qwen3_5.modeling_qwen3_5 as mq

OUT = os.path.join(os.environ['PROJ'], 'logs/0001/p5')
repo = 'Qwen/Qwen3.5-0.8B'
img = os.path.join(os.environ['PROJ'], 'tmp/coco/000000039769.jpg')
log_stream = io.StringIO()
hd = logging.StreamHandler(log_stream); hd.setLevel(logging.WARNING)
transformers.logging.get_logger().addHandler(hd)
warnings.simplefilter('always')

import inspect
def impl_of(name):
    w = getattr(mq, name)
    impl = inspect.getclosurevars(w).nonlocals.get('implementation')
    return f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__qualname__', '?')}"
R = {'mode': mode, 'impl': {n: impl_of(n) for n in ['torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
                                                     'causal_conv1d_fn', 'causal_conv1d_update']}}
proc = AutoProcessor.from_pretrained(repo)
msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe the image in one sentence.'}, {'type': 'image', 'path': img}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = proc(text=[text], images=[__import__('PIL.Image', fromlist=['Image']).open(img).convert('RGB')], return_tensors='pt')
R['seq_len'] = enc['input_ids'].shape[1]
R['image_grid_thw'] = enc['image_grid_thw'].tolist()

orig = {n: getattr(mq, n) for n in ['torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule']}
def set_torch(on):
    for n, f in orig.items():
        setattr(mq, n, f.__wrapped__ if on else f)

def stats(a, b):
    d = (a - b).abs()
    return {'max_abs': d.max().item(), 'mean_abs': d.mean().item(), 'rel_fro': (d.norm() / b.norm()).item(),
            'argmax_agree_frac': (a.argmax(-1) == b.argmax(-1)).float().mean().item(),
            'last_pos_argmax_equal': bool(a[0, -1].argmax() == b[0, -1].argmax()),
            'logit_abs_max': b.abs().max().item()}

def sha(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

for dtype in [torch.bfloat16, torch.float32]:
    dn = str(dtype).split('.')[-1]
    log_stream.truncate(0); log_stream.seek(0)
    with warnings.catch_warnings(record=True) as wl:
        model = AutoModelForImageTextToText.from_pretrained(repo, dtype=dtype, device_map='cuda').eval()
        R[f'{dn}_attn_implementation'] = model.config._attn_implementation
        inp = {k: v.to('cuda') for k, v in enc.items()}
        with torch.no_grad():
            if mode == 'inproc':
                set_torch(False)
                a = model(**inp).logits.float()
                a2 = model(**inp).logits.float()
                set_torch(True)
                b = model(**inp).logits.float()
                b2 = model(**inp).logits.float()
                set_torch(False)
                R[f'{dn}_fla_vs_torchref'] = stats(a, b)
                R[f'{dn}_fla_repeat_bitwise_equal'] = bool(torch.equal(a, a2))
                R[f'{dn}_torchref_repeat_bitwise_equal'] = bool(torch.equal(b, b2))
                R[f'{dn}_sha_fla'] = sha(a); R[f'{dn}_sha_torchref'] = sha(b)
                R[f'{dn}_last_logits_top5_fla'] = torch.topk(a[0, -1], 5).indices.tolist()
                R[f'{dn}_last_logits_top5_torchref'] = torch.topk(b[0, -1], 5).indices.tolist()
            else:
                b = model(**inp).logits.float()
                R[f'{dn}_sha_fallback'] = sha(b)
                R[f'{dn}_last_logits_top5_fallback'] = torch.topk(b[0, -1], 5).indices.tolist()
    R[f'{dn}_warnings'] = sorted(set(str(w.message)[:250] for w in wl))
    R[f'{dn}_log'] = log_stream.getvalue()[-2000:]
    del model; torch.cuda.empty_cache()
json.dump(R, open(os.path.join(OUT, f'gdn_model_paths_{mode}.json'), 'w'), indent=1)
print(json.dumps(R, indent=1))
