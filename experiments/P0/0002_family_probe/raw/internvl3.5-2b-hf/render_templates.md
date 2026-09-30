| 调用方式 | 生成提示（原样，截取末段） | 备注 |
|---|---|---|
| default(no kwarg) \| text | `'<\|im_start\|>user\nWhat is 2+3?<\|im_end\|>\n<\|im_start\|>assistant\n'` |  |
| default(no kwarg) \| image | `'<\|im_start\|>user\n<IMG_CONTEXT>\nDescribe the image.<\|im_end\|>\n<\|im_start\|>assistant\n'` |  |
| enable_thinking=False \| text | `'<\|im_start\|>user\nWhat is 2+3?<\|im_end\|>\n<\|im_start\|>assistant\n'` | Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |
| enable_thinking=False \| image | `'<\|im_start\|>user\n<IMG_CONTEXT>\nDescribe the image.<\|im_end\|>\n<\|im_start\|>assistant\n'` | Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |
| enable_thinking=True \| text | `'<\|im_start\|>user\nWhat is 2+3?<\|im_end\|>\n<\|im_start\|>assistant\n'` | Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |
| enable_thinking=True \| image | `'<\|im_start\|>user\n<IMG_CONTEXT>\nDescribe the image.<\|im_end\|>\n<\|im_start\|>assistant\n'` | Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |
| chat_template_kwargs={enable_thinking:False} \| text | `'<\|im_start\|>user\nWhat is 2+3?<\|im_end\|>\n<\|im_start\|>assistant\n'` | Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |
| chat_template_kwargs={enable_thinking:False} \| image | `'<\|im_start\|>user\n<IMG_CONTEXT>\nDescribe the image.<\|im_end\|>\n<\|im_start\|>assistant\n'` | Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |
| system=R1_SYSTEM_PROMPT (card thinking mode) \| text | `'…he query. The final answer should be standalone and not reference the thinking section.<\|im_end\|>\n<\|im_start\|>user\nWhat is 2+3?<\|im_end\|>\n<\|im_start\|>assistant\n'` |  |
| system=R1_SYSTEM_PROMPT (card thinking mode) \| image | `'…nswer should be standalone and not reference the thinking section.<\|im_end\|>\n<\|im_start\|>user\n<IMG_CONTEXT>\nDescribe the image.<\|im_end\|>\n<\|im_start\|>assistant\n'` |  |