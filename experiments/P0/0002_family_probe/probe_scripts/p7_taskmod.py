"""RUN-0002 §3.6 (D1 instrument) for Gemma 4 / InternVL3.5-HF: hidden states at image-token positions under two
instructions of equal token length. image_first -> expect torch.equal at every layer; instruction_first -> layer 0 equal, later differ.
batch=1, bf16, default kernels (sdpa), default processor resolution. Same image as RUN-0001 (COCO 000000039769).
usage: python p7_taskmod.py gemma|internvl <repo> <rev> <tag>
"""
import os, sys, json, torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

fam, repo, rev, tag = sys.argv[1:5]
OUT = os.path.join(os.environ['PROJ'], 'logs/0002', tag)
img = Image.open(os.path.join(os.environ['PROJ'], 'tmp/coco/000000039769.jpg')).convert('RGB')
proc = AutoProcessor.from_pretrained(repo, revision=rev)
tok = proc.tokenizer
CANDS = [('Where is the cat in this image?', 'What color is the sofa here?'),
         ('Describe the cat.', 'Describe the sofa.'),
         ('Count the remotes.', 'Count the pillows.')]
pair = None
for a, b in CANDS:
    if len(tok(a, add_special_tokens=False)['input_ids']) == len(tok(b, add_special_tokens=False)['input_ids']):
        pair = (a, b); break
assert pair is not None
model = AutoModelForImageTextToText.from_pretrained(repo, revision=rev, dtype=torch.bfloat16, device_map='cuda').eval()
img_id = proc.image_token_id
extra = {'enable_thinking': False} if fam == 'gemma' else {}


def run(order, instr):
    content = [{'type': 'image'}, {'type': 'text', 'text': instr}] if order == 'image_first' else \
              [{'type': 'text', 'text': instr}, {'type': 'image'}]
    text = proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True, **extra)
    enc = proc(text=[text], images=[img], return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    pos = (enc['input_ids'][0] == img_id).nonzero().squeeze(1)
    return {'text': text, 'ids': enc['input_ids'][0].tolist(), 'img_pos': pos.tolist(), 'hs': [h[0, pos].detach().clone() for h in out.hidden_states]}


R = {'repo': repo, 'revision': rev, 'dtype': 'bf16', 'instructions': pair,
     'instr_token_lens': [len(tok(x, add_special_tokens=False)['input_ids']) for x in pair],
     'attn_implementation': model.config._attn_implementation}
if fam == 'gemma':
    tc = model.config.get_text_config()
    R['structure_facts'] = {'sliding_window': tc.sliding_window, 'num_kv_shared_layers': tc.num_kv_shared_layers,
                            'use_bidirectional_attention': tc.use_bidirectional_attention, 'layer_types': tc.layer_types}
    R['card_quote_modality_order'] = 'For optimal performance with multimodal inputs, place: Image content **before** the text in your prompt.'
for order in ['image_first', 'instruction_first']:
    A, B = run(order, pair[0]), run(order, pair[1])
    assert A['img_pos'] == B['img_pos']
    rows = []
    for li, (ha, hb) in enumerate(zip(A['hs'], B['hs'])):
        d = (ha.float() - hb.float()).abs()
        ct = F.cosine_similarity(ha.float(), hb.float(), dim=-1)
        rows.append({'layer': li, 'equal': bool(torch.equal(ha, hb)), 'max_abs_delta': d.max().item(), 'mean_abs_delta': d.mean().item(),
                     'cos_flat': F.cosine_similarity(ha.float().flatten(), hb.float().flatten(), dim=0).item(),
                     'cos_min_token': ct.min().item(), 'cos_mean_token': ct.mean().item(), 'n_tokens_changed': int((d.amax(-1) > 0).sum())})
    R[order] = {'n_image_tokens': len(A['img_pos']), 'img_pos_range': [A['img_pos'][0], A['img_pos'][-1]], 'seq_len': len(A['ids']),
                'rendered_A': A['text'], 'rendered_B': B['text'], 'layers': rows}
    print(f'== {order} n_img_tok={len(A["img_pos"])} seq_len={len(A["ids"])} img_pos={A["img_pos"][0]}..{A["img_pos"][-1]}')
    for r in rows:
        print(f"  L{r['layer']:02d} equal={r['equal']!s:5s} max|d|={r['max_abs_delta']:.4e} cos_flat={r['cos_flat']:.6f} cos_min_tok={r['cos_min_token']:.6f} changed_tok={r['n_tokens_changed']}")
json.dump(R, open(os.path.join(OUT, 'taskmod_bf16.json'), 'w'), indent=1, ensure_ascii=False)
