"""LLMSeg 外部基线（DEV-0066；D125-4）：官方代码（third_party/llmseg，固定在 f2191e66）+ 本地 Qwen3.5 批量提供方 + W2 口径的评测。

- vendor.py：引入的官方文件、出处核对（PROVENANCE.md 的 sha256）与加载；
- provider.py：本地提供方（经我们的适配器，关思考、贪心）与锁步合批的调度器，通过官方 llm.py 的接口注入（不改它的文件）；
- runner.py：W2 test 子集（hash_rank(salt = w2_llmseg_v1) 前 300 句）、逐句跑官方管线、W2 同口径计分与输出、与 W2-c 的配对比较。
"""
