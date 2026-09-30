"""RUN-0002 §3.7: bf16 peak memory + HF generate throughput for Gemma 4 E2B-it / InternVL3.5-2B-HF (same protocol as RUN-0001 p5_8_mem.py).
 image: the two COCO val2017 photos resized to 1024x1024
   gemma:    max_soft_tokens=1120 -> 1089 visual tokens (closest budget to ~1K)
   internvl: crop_to_patches=True, max_patches=4 -> 2x2 tiles + thumbnail = 5 tiles = 1280 visual tokens (closest setting to ~1K for a square image)
 infer: forward bs1/bs16 (logits_to_keep=1 and full logits); generate bs16 max_new_tokens=64 greedy thinking off
 lora:  r=32 on every nn.Linear inside model.model.language_model, gradient checkpointing, bs8, one fwd+bwd (no optimizer step, nothing saved)
usage: python p7_mem.py gemma|internvl <repo> <rev> <tag> infer|lora
"""
import os, sys, json, time, gc, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

fam, repo, rev, tag, mode = sys.argv[1:6]
OUT = os.path.join(os.environ['PROJ'], 'logs/0002', tag)
COCO = os.path.join(os.environ['PROJ'], 'tmp/coco')
imgs = [Image.open(os.path.join(COCO, f)).convert('RGB').resize((1024, 1024), Image.BICUBIC) for f in ['000000039769.jpg', '000000000139.jpg']]
proc = AutoProcessor.from_pretrained(repo, revision=rev)
proc.tokenizer.padding_side = 'left'
GB = 1024 ** 3
IMG_KW = {'gemma': {'max_soft_tokens': 1120}, 'internvl': {'crop_to_patches': True, 'max_patches': 4}}[fam]
extra = {'enable_thinking': False} if fam == 'gemma' else {}
END = {'gemma': '<turn|>\n', 'internvl': '<|im_end|>\n'}[fam]


def batch(n, prompt='Describe the image in detail.', answer=None):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra)
    if answer is not None:
        text = text + answer + END
    ims = [imgs[i % 2] for i in range(n)]
    if fam == 'gemma':
        ims = [[im] for im in ims]  # Gemma4Processor expects one image list per sample in a batch
    return proc(text=[text] * n, images=ims, return_tensors='pt', padding=True, **IMG_KW).to('cuda')


def mem_reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()


def mem_read():
    torch.cuda.synchronize()
    return {'peak_alloc_GiB': round(torch.cuda.max_memory_allocated() / GB, 3), 'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / GB, 3)}


R = {'repo': repo, 'revision': rev, 'mode': mode, 'gpu': torch.cuda.get_device_name(0), 'image_kwargs': IMG_KW}
mem_reset()
model = AutoModelForImageTextToText.from_pretrained(repo, revision=rev, dtype=torch.bfloat16, device_map='cuda')
R['after_load'] = {'alloc_GiB': round(torch.cuda.memory_allocated() / GB, 3), **mem_read()}
R['attn_implementation'] = model.config._attn_implementation
img_id = proc.image_token_id
if mode == 'infer':
    model.eval()
    e1 = batch(1)
    R['seq_len'] = int(e1['input_ids'].shape[1]); R['n_visual_tokens'] = int((e1['input_ids'][0] == img_id).sum())
    for bs in [1, 16]:
        enc = batch(bs)
        for ltk, nm in [(1, 'logits_to_keep=1'), (0, 'full_logits')]:
            mem_reset()
            with torch.no_grad():
                t0 = time.time(); out = model(**enc, logits_to_keep=ltk, use_cache=False); torch.cuda.synchronize(); dt = time.time() - t0
            R[f'forward_bs{bs}_{nm}'] = {**mem_read(), 'wall_s': round(dt, 3), 'logits_shape': list(out.logits.shape)}
            del out
    enc = batch(16)
    gk = dict(do_sample=False, max_new_tokens=64, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        model.generate(**enc, **gk)
        runs = []
        for _ in range(3):
            mem_reset(); t0 = time.time()
            out = model.generate(**enc, **gk); torch.cuda.synchronize(); dt = time.time() - t0
            new = out[:, enc['input_ids'].shape[1]:]
            runs.append({'wall_s': round(dt, 3), 'samples_per_s': round(16 / dt, 3), 'new_tokens_matrix': list(new.shape), **mem_read()})
    R['generate_bs16_max_new_64'] = {'gen_kwargs': {k: v for k, v in gk.items() if v is not None}, 'thinking': 'off', 'prompt_len': int(enc['input_ids'].shape[1]),
                                    'runs': runs, 'sample0_text': proc.tokenizer.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=False)[:400]}
elif mode == 'lora':
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    enc = batch(8, answer='Two cats are lying on a pink couch next to two remote controls.')
    labels = enc['input_ids'].clone(); labels[enc['attention_mask'] == 0] = -100
    R['seq_len'] = int(enc['input_ids'].shape[1])
    targets = sorted(n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith('model.language_model.'))
    R['lora_target_suffix_counts'] = {}
    for n in targets:
        k = n.split('.')[-1]; R['lora_target_suffix_counts'][k] = R['lora_target_suffix_counts'].get(k, 0) + 1
    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=targets, task_type='CAUSAL_LM')
    pm = get_peft_model(model, cfg)
    pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    pm.enable_input_require_grads()
    pm.train()
    n_tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    n_mod = sum(1 for n, m in pm.named_modules() if hasattr(m, 'lora_A') and len(getattr(m, 'lora_A', {})) > 0)
    for rep in range(2):  # rep 0 includes JIT/warmup; rep 1 is the steady-state step
        mem_reset(); t0 = time.time()
        out = pm(**enc, labels=labels); out.loss.backward(); torch.cuda.synchronize(); dt = time.time() - t0
        R[f'lora_r32_lm_linear_bs8_gc_rep{rep}'] = {**mem_read(), 'wall_s_fwd_bwd': round(dt, 3), 'loss': out.loss.item(), 'trainable_params': n_tr,
                                                    'n_lora_modules': n_mod,
                                                    'grad_nonzero_lora_B': any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in pm.named_parameters() if 'lora_B' in n)}
        del out; pm.zero_grad(set_to_none=True)
json.dump(R, open(os.path.join(OUT, f'mem_{mode}.json'), 'w'), indent=1)
print(json.dumps(R, indent=1))
