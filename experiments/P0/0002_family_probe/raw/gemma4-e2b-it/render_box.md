提示词 `detect {obj}`，出处：https://ai.google.dev/gemma/docs/capabilities/vision/image (object detection example text "detect person and cat")；生成设置 {"do_sample": false, "max_new_tokens": 256, "enable_thinking": false}。

- `000000039769.jpg`（640×480，视觉 token 266），目标 "the cat on the left"，生成 46 token：`'```json\n[\n  {"box_2d": [107, 13, 987, 496], "label": "the cat on the left"}\n]\n```<turn|>'`
- `000000000139.jpg`（640×426，视觉 token 260），目标 "the television"，生成 43 token：`'```json\n[\n  {"box_2d": [394, 13, 621, 244], "label": "the television"}\n]\n```<turn|>'`