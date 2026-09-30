"""RUN-0001 §5.7 (D1 instrument): hidden states at image-token positions under two instructions of equal token length.
 image_first:       [image, instr]  -> expect torch.equal at every layer (exact-zero instrument)
 instruction_first: [instr, image]  -> expect layer 0 equal, later layers differ
batch=1 per run, same dtype, same kernels (default transformers path: fla chunk GDN + sdpa).
usage: python p5_7_taskmod.py <repo> <tag> <dtype: bf16|fp32>
"""
import os, sys, json, torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

repo, tag, dts = sys.argv[1], sys.argv[2], sys.argv[3]
dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[dts]
OUT = os.path.join(os.environ['PROJ'], 'logs/0001/p5', tag)
os.makedirs(OUT, exist_ok=True)
img = Image.open(os.path.join(os.environ['PROJ'], 'tmp/coco/000000039769.jpg')).convert('RGB')
proc = AutoProcessor.from_pretrained(repo)
tok = proc.tokenizer
# two instructions with identical token length, different content
CANDS = [('Where is the cat in this image?', 'What color is the sofa here?'),
         ('Describe the cat.', 'Describe the sofa.'),
         ('Count the remotes.', 'Count the pillows.')]
pair = None
for a, b in CANDS:
    la, lb = len(tok(a, add_special_tokens=False)['input_ids']), len(tok(b, add_special_tokens=False)['input_ids'])
    if la == lb:
        pair = (a, b); break
assert pair is not None, 'no equal-length pair'
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=dtype, device_map='cuda').eval()
img_id = tok.convert_tokens_to_ids('<|image_pad|>')


def run(order, instr):
    content = [{'type': 'image'}, {'type': 'text', 'text': instr}] if order == 'image_first' else \
              [{'type': 'text', 'text': instr}, {'type': 'image'}]
    msgs = [{'role': 'user', 'content': content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = proc(text=[text], images=[img], return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    pos = (enc['input_ids'][0] == img_id).nonzero().squeeze(1)
    hs = [h[0, pos].detach().clone() for h in out.hidden_states]
    return {'text': text, 'ids': enc['input_ids'][0].tolist(), 'img_pos': pos.tolist(), 'hs': hs}


R = {'repo': repo, 'dtype': dts, 'instructions': pair,
     'instr_token_lens': [len(tok(x, add_special_tokens=False)['input_ids']) for x in pair],
     'attn_implementation': model.config._attn_implementation}
for order in ['image_first', 'instruction_first']:
    A, B = run(order, pair[0]), run(order, pair[1])
    assert A['img_pos'] == B['img_pos'], 'image positions differ between the two instructions'
    rows = []
    for li, (ha, hb) in enumerate(zip(A['hs'], B['hs'])):
        d = (ha.float() - hb.float()).abs()
        cos_tok = F.cosine_similarity(ha.float(), hb.float(), dim=-1)
        rows.append({'layer': li, 'equal': bool(torch.equal(ha, hb)), 'max_abs_delta': d.max().item(),
                     'mean_abs_delta': d.mean().item(),
                     'cos_flat': F.cosine_similarity(ha.float().flatten(), hb.float().flatten(), dim=0).item(),
                     'cos_min_token': cos_tok.min().item(), 'cos_mean_token': cos_tok.mean().item(),
                     'n_tokens_changed': int((d.amax(-1) > 0).sum())})
    R[order] = {'n_image_tokens': len(A['img_pos']), 'img_pos_range': [A['img_pos'][0], A['img_pos'][-1]],
                'seq_len': len(A['ids']), 'rendered_A': A['text'], 'rendered_B': B['text'], 'layers': rows}
    print(f'== {order}  n_img_tok={len(A["img_pos"])} seq_len={len(A["ids"])}')
    for r in rows:
        print(f"  L{r['layer']:02d} equal={r['equal']!s:5s} max|d|={r['max_abs_delta']:.4e} cos_flat={r['cos_flat']:.6f} "
              f"cos_min_tok={r['cos_min_token']:.6f} changed_tok={r['n_tokens_changed']}")
json.dump(R, open(os.path.join(OUT, f'taskmod_{dts}.json'), 'w'), indent=1, ensure_ascii=False)
