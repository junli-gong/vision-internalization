| 调用方式 | 生成提示（原样，截取末段） | 备注 |
|---|---|---|
| default(no kwarg) \| text | `'<bos><\|turn>user\nWhat is 2+3?<turn\|>\n<\|turn>model\n'` |  |
| default(no kwarg) \| image | `'<bos><\|turn>user\n<\|image\|>Describe the image.<turn\|>\n<\|turn>model\n'` |  |
| enable_thinking=False \| text | `'<bos><\|turn>user\nWhat is 2+3?<turn\|>\n<\|turn>model\n'` |  |
| enable_thinking=False \| image | `'<bos><\|turn>user\n<\|image\|>Describe the image.<turn\|>\n<\|turn>model\n'` |  |
| enable_thinking=True \| text | `'<bos><\|turn>system\n<\|think\|>\n<turn\|>\n<\|turn>user\nWhat is 2+3?<turn\|>\n<\|turn>model\n'` |  |
| enable_thinking=True \| image | `'<bos><\|turn>system\n<\|think\|>\n<turn\|>\n<\|turn>user\n<\|image\|>Describe the image.<turn\|>\n<\|turn>model\n'` |  |
| chat_template_kwargs={enable_thinking:False} \| text | `'<bos><\|turn>user\nWhat is 2+3?<turn\|>\n<\|turn>model\n'` | Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |
| chat_template_kwargs={enable_thinking:False} \| image | `'<bos><\|turn>user\n<\|image\|>Describe the image.<turn\|>\n<\|turn>model\n'` | Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs` |