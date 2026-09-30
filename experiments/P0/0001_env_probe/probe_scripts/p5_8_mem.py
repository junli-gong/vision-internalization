"""RUN-0001 §5.8: measured peak GPU memory and HF generate throughput (replaces brief v0.1 §5.4 estimates).
usage: python p5_8_mem.py <repo> <tag> infer|lora
 infer: (a) bf16 forward, ~1K visual tokens, batch 1/16, full logits vs logits_to_keep=1
        (b) online-probe forward: output_hidden_states=True (all layers), batch 8
        (c) HF generate throughput: batch 16, max_new_tokens=64, thinking off, greedy
 lora:  LoRA r=32 on linear layers, gradient checkpointing, batch 8, one forward+backward (no optimizer step, nothing saved)
        two target sets: peft 'all-linear' (includes vision tower) and language-model-only linear layers
Image: COCO val2017 photos resized to 1024x1024 -> 32x32 = 1024 visual tokens.
"""
import os, sys, json, time, gc, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

repo, tag, mode = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = os.path.join(os.environ['PROJ'], 'logs/0001/p5', tag)
os.makedirs(OUT, exist_ok=True)
COCO = os.path.join(os.environ['PROJ'], 'tmp/coco')
imgs = [Image.open(os.path.join(COCO, f)).convert('RGB').resize((1024, 1024), Image.BICUBIC)
        for f in ['000000039769.jpg', '000000000139.jpg']]
proc = AutoProcessor.from_pretrained(repo)
proc.tokenizer.padding_side = 'left'
GB = 1024 ** 3


def batch(n, prompt='Describe the image in detail.', answer=None):
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if answer is not None:
        text = text + answer + '<|im_end|>\n'
    return proc(text=[text] * n, images=[imgs[i % 2] for i in range(n)], return_tensors='pt', padding=True).to('cuda')


def mem_reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()


def mem_read():
    torch.cuda.synchronize()
    return {'peak_alloc_GiB': round(torch.cuda.max_memory_allocated() / GB, 3),
            'peak_reserved_GiB': round(torch.cuda.max_memory_reserved() / GB, 3)}


R = {'repo': repo, 'mode': mode, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
mem_reset()
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map='cuda')
R['after_load'] = {'alloc_GiB': round(torch.cuda.memory_allocated() / GB, 3), **mem_read()}
R['attn_implementation'] = model.config._attn_implementation

if mode == 'infer':
    model.eval()
    enc1 = batch(1)
    R['seq_len'] = int(enc1['input_ids'].shape[1]); R['n_visual_tokens'] = int((enc1['input_ids'] == proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum())
    R['image_grid_thw'] = enc1['image_grid_thw'][0].tolist()
    # (a) forward
    for bs in [1, 16]:
        enc = batch(bs)
        for ltk, name in [(0, 'full_logits'), (1, 'logits_to_keep=1')]:
            mem_reset()
            with torch.no_grad():
                t0 = time.time(); out = model(**enc, logits_to_keep=ltk, use_cache=False); torch.cuda.synchronize(); dt = time.time() - t0
            R[f'forward_bs{bs}_{name}'] = {**mem_read(), 'wall_s': round(dt, 3), 'logits_shape': list(out.logits.shape)}
            del out
    # (b) online probe forward
    enc = batch(8)
    for ltk, name in [(1, 'logits_to_keep=1'), (0, 'full_logits')]:
        mem_reset()
        with torch.no_grad():
            t0 = time.time(); out = model(**enc, output_hidden_states=True, logits_to_keep=ltk, use_cache=False); torch.cuda.synchronize(); dt = time.time() - t0
        R[f'probe_forward_bs8_all_hidden_{name}'] = {**mem_read(), 'wall_s': round(dt, 3), 'n_hidden_states': len(out.hidden_states),
                                                     'hidden_shape': list(out.hidden_states[-1].shape),
                                                     'hidden_bytes_all_layers_GiB': round(sum(h.numel() * h.element_size() for h in out.hidden_states) / GB, 3)}
        del out
    # (c) generate throughput
    enc = batch(16)
    gk = dict(do_sample=False, max_new_tokens=64, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        model.generate(**enc, **gk)  # warmup
        runs = []
        for _ in range(3):
            mem_reset(); torch.cuda.synchronize(); t0 = time.time()
            out = model.generate(**enc, **gk); torch.cuda.synchronize(); dt = time.time() - t0
            new = out[:, enc['input_ids'].shape[1]:]
            eos = proc.tokenizer.convert_tokens_to_ids('<|endoftext|>')
            n_gen = [int((row != eos).sum()) for row in new]  # tokens before padding with eos
            runs.append({'wall_s': round(dt, 3), 'samples_per_s': round(16 / dt, 3), 'new_tokens_matrix': list(new.shape),
                         'gen_tokens_per_sample_minmax': [min(n_gen), max(n_gen)], **mem_read()})
    R['generate_bs16_max_new_64'] = {'gen_kwargs': {k: v for k, v in gk.items() if v is not None}, 'enable_thinking': False,
                                    'prompt_len': int(enc['input_ids'].shape[1]), 'runs': runs,
                                    'sample0_text': proc.tokenizer.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=False)[:400]}
elif mode == 'lora':
    from peft import LoraConfig, get_peft_model
    import peft
    R['peft'] = peft.__version__
    enc = batch(8, answer='Two cats are lying on a pink couch next to two remote controls.')
    labels = enc['input_ids'].clone(); labels[enc['attention_mask'] == 0] = -100
    R['seq_len'] = int(enc['input_ids'].shape[1])
    base_state = None
    for tname in ['all-linear', 'language_model_linear_only']:
        if tname == 'all-linear':
            tm = 'all-linear'
        else:
            tm = r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$'
        cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=tm, task_type='CAUSAL_LM')
        pm = get_peft_model(model, cfg)
        pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        pm.enable_input_require_grads()
        pm.train()
        n_tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        n_mod = sum(1 for n, m in pm.named_modules() if hasattr(m, 'lora_A') and len(getattr(m, 'lora_A', {})) > 0)
        n_vis = sum(1 for n, m in pm.named_modules() if hasattr(m, 'lora_A') and len(getattr(m, 'lora_A', {})) > 0 and '.visual.' in n)
        mem_reset()
        t0 = time.time()
        out = pm(**enc, labels=labels)
        out.loss.backward(); torch.cuda.synchronize(); dt = time.time() - t0
        R[f'lora_r32_{tname}_bs8_gc'] = {**mem_read(), 'wall_s_fwd_bwd': round(dt, 3), 'loss': out.loss.item(),
                                         'trainable_params': n_tr, 'n_lora_modules': n_mod, 'n_lora_modules_in_visual': n_vis,
                                         'grad_nonzero_lora_B': any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in pm.named_parameters() if 'lora_B' in n)}
        del out
        pm.zero_grad(set_to_none=True)
        model = pm.unload()  # strip LoRA (not merged), back to base for the next config
        model.gradient_checkpointing_disable()
        mem_reset()
json.dump(R, open(os.path.join(OUT, f'mem_{mode}.json'), 'w'), indent=1)
print(json.dumps(R, indent=1))
