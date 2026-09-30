"""RUN-0001 §5.5: zero-shot 'give the box' on 2 COCO val2017 images. Greedy, thinking OFF (explicit), one fixed prompt, no tuning.
usage: python p5_5_box_zeroshot.py <repo> <tag>
"""
import os, sys, json, time, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

repo, tag = sys.argv[1], sys.argv[2]
OUT = os.path.join(os.environ['PROJ'], 'logs/0001/p5', tag)
COCO = os.path.join(os.environ['PROJ'], 'tmp/coco')
CASES = [('000000039769.jpg', 'the cat on the left'), ('000000000139.jpg', 'the television')]
PROMPT = 'Locate {obj} in the image. Report bbox coordinates in JSON format.'  # wording follows the Qwen3-VL 2d_grounding cookbook

proc = AutoProcessor.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map='cuda').eval()
res = {'repo': repo, 'prompt_template': PROMPT, 'generation': {'do_sample': False, 'max_new_tokens': 256, 'enable_thinking': False},
       'model_generation_config_eos': model.generation_config.eos_token_id, 'cases': []}
for fn, obj in CASES:
    img = Image.open(os.path.join(COCO, fn)).convert('RGB')
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT.format(obj=obj)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = proc(text=[text], images=[img], return_tensors='pt').to('cuda')
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=256, temperature=None, top_p=None, top_k=None)
    dt = time.time() - t0
    new = out[0, enc['input_ids'].shape[1]:]
    thw = enc['image_grid_thw'][0].tolist()
    res['cases'].append({'image': fn, 'image_wh': img.size, 'object': obj, 'rendered_prompt': text,
                         'image_grid_thw': thw, 'resized_hw_px': [thw[1] * 16, thw[2] * 16],
                         'n_new_tokens': int(new.shape[0]), 'last_token_ids': new[-3:].tolist(),
                         'raw_output_with_special': proc.tokenizer.decode(new, skip_special_tokens=False),
                         'wall_s': round(dt, 3)})
    print(fn, obj, '->', repr(res['cases'][-1]['raw_output_with_special']))
json.dump(res, open(os.path.join(OUT, 'box_zeroshot.json'), 'w'), indent=1, ensure_ascii=False)
