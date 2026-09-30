"""RUN-0002 §3.5: zero-shot localization on the 2 RUN-0001 COCO val2017 images, same targets. Greedy, thinking off, one fixed prompt per family.
 gemma:    prompt pattern from ai.google.dev/gemma/docs/capabilities/vision/image object-detection example: "detect person and cat" -> "detect {obj}"
 internvl: prompt from internvl.readthedocs.io chat_data_format RefCOCO example:
           "Please provide the bounding box coordinate of the region this sentence describes: <ref>{obj}</ref>"
Image placed before text in both (Gemma card: image before text; InternVL example: "<image>\\n" first). Default processor resolution settings.
usage: python p7_box.py gemma|internvl <repo> <rev> <tag>
"""
import os, sys, json, time, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

fam, repo, rev, tag = sys.argv[1:5]
OUT = os.path.join(os.environ['PROJ'], 'logs/0002', tag)
COCO = os.path.join(os.environ['PROJ'], 'tmp/coco')
CASES = [('000000039769.jpg', 'the cat on the left'), ('000000000139.jpg', 'the television')]
PROMPT = {'gemma': 'detect {obj}',
          'internvl': 'Please provide the bounding box coordinate of the region this sentence describes: <ref>{obj}</ref>'}[fam]
SRC = {'gemma': 'https://ai.google.dev/gemma/docs/capabilities/vision/image (object detection example text "detect person and cat")',
       'internvl': 'https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html (RefCOCO grounding example)'}[fam]
proc = AutoProcessor.from_pretrained(repo, revision=rev)
model = AutoModelForImageTextToText.from_pretrained(repo, revision=rev, dtype=torch.bfloat16, device_map='cuda').eval()
extra = {'enable_thinking': False} if fam == 'gemma' else {}
res = {'repo': repo, 'revision': rev, 'prompt_template': PROMPT, 'prompt_source': SRC,
       'generation': {'do_sample': False, 'max_new_tokens': 256, **({'enable_thinking': False} if fam == 'gemma' else {'thinking': 'off (no R1 system prompt)'})},
       'eos_token_id': model.generation_config.eos_token_id, 'cases': []}
for fn, obj in CASES:
    img = Image.open(os.path.join(COCO, fn)).convert('RGB')
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT.format(obj=obj)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra)
    enc = proc(text=[text], images=[img], return_tensors='pt').to('cuda')
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=256, temperature=None, top_p=None, top_k=None)
    dt = time.time() - t0
    new = out[0, enc['input_ids'].shape[1]:]
    c = {'image': fn, 'image_wh': list(img.size), 'object': obj, 'rendered_prompt': text,
         'n_image_tokens': int((enc['input_ids'][0] == proc.image_token_id).sum()), 'n_new_tokens': int(new.shape[0]),
         'last_token_ids': new[-3:].tolist(), 'raw_output_with_special': proc.tokenizer.decode(new, skip_special_tokens=False),
         'wall_s': round(dt, 3)}
    if fam == 'internvl':
        c['n_tiles'] = int(enc['pixel_values'].shape[0])
    res['cases'].append(c)
    print(fn, obj, '->', repr(c['raw_output_with_special']))
json.dump(res, open(os.path.join(OUT, 'box_zeroshot.json'), 'w'), indent=1, ensure_ascii=False)
