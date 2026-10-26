# third_party/llmseg 的出处（DEV-0066；D125-4）

- 来源：https://github.com/ajhamdi/llmseg-code
- commit：`f2191e6605e40a5be5d932187fc055708ae098cf`（2026-10-22T10:09:44+03:00，"Add arXiv link and BibTeX, News and Examples to README"）
- 论文：arXiv 2610.29409，"LLMSeg: Training-Free Text-Prompted Image Segmentation with B-Splines"（Abdullah Hamdi，KAUST）
- 许可证：MIT（`LICENSE` 原样保留）。取法：在开发机的临时目录 `git clone` 后 `git checkout` 上面的 commit，逐个文件原样拷贝（不在它的目录里运行任何东西）；
  下表的 sha256 在拷贝前后各算一次，相同。

## 文件与 sha256

只放运行管线必需的文件：方法本身 4 个文件，加上它自己的离线测试与该测试 import 的两个评测模块（`tests/test_offline.py` 要 `evals/common.py`、
`evals/gtmask.py`）。没有放 toy 图、评测数据、下载脚本、其余评测脚本、README 与图片。

| 文件 | sha256 |
|---|---|
| `LICENSE` | `c4781a2606ae22d8afb98f76a150dea46584184bd9f0e7e0c071261386b26401` |
| `llm.py` | `e68e315656b723a5a75545b4ddd5a7b75caabea60da25326ede74829fea0552f` |
| `pipeline.py` | `a467455548d2a5539d864e49f94031f00f85af04151726bb760b73099130337c` |
| `prompts.py` | `184d331447bd3c4eb487c183a6f5a531c9d46e66d9db6ca506ea57daf796a5e1` |
| `geometry.py` | `f41d41ef55fcfb8a75b58483befd7e8b31ff7d936e1ee29375acf90530cfd88f` |
| `tests/test_offline.py` | `66f7d84004aff82430e7866a99df26280c7c4a1be37d7795c79cab0014f12c51` |
| `evals/common.py` | `3f57930a770cb701bddacc326c900edaada23f67f7afdb4980f05242362196ee` |
| `evals/gtmask.py` | `144c0f8237e5a45c759a873d729283498f618afc328abf3b9fec80c343ba99dd` |

`src/llmseg/vendor.check_vendor()` 每次运行前现算并与这张表逐项比较（`tests/test_llmseg_vendor.py` 也核对）；运行身份（config.yaml 的
`llmseg.code`）记 commit、本文件的 sha256 与逐文件 sha256。

## 我们改了什么

**官方文件一个字节都没改。** 本地提供方全部在我们这边（`src/llmseg/provider.py`），运行时只替换官方模块的属性（`provider.install()`，
`restore()` 还原）：

| 官方的属性 | 换成 | 为什么 |
|---|---|---|
| `llm._complete(provider, model, prompt, path, timeout)` | `LocalProvider.complete`：经我们的适配器生成（关思考、贪心），锁步合批 | 官方只支持四家闭源 API；这是官方离线测试自己用的接缝（`tests/test_offline.py` 也是换它） |
| `llm._call_gemini`、`llm._call_anthropic`、`llm._call_openai_compatible` | 抛 `ExternalCallForbidden`（BaseException） | 不用任何闭源 API：即使环境里有 key 也到不了网络 |
| `llm.configure(provider, timeout)`（官方公开函数，调用而非替换） | `provider="local"`、`timeout=7 天` | provider 名只进日志；合批时一次调用要排队，超时交给我们的调度器 |
| `llm.log`、`pipeline.log`、`geometry.log` | 按线程分流的日志（`LogRouter`），写进每条样本的 `run.log` | 官方评测每条一个子进程、stderr 进 `run.log`；我们为了合批在一个进程里多线程跑 |

调用方式的不同（不是改它的代码）：
- 官方评测（`evals/common.run_pipeline`）每条样本起一个 `pipeline.py` 子进程；我们在本进程里直接调 `pipeline.run(...)`，参数与它的命令行主配置相同
  （`--iterations 8`，`--topology`、`--refine-box`、`--no-postprocess` 都关，`--max-instances` 取 `pipeline.MAX_INSTANCES`）。
- mask 的取法同 `evals/common.run_pipeline` 与 `load_pred`：修正后 = `rec_mask.png`（没有退回 `seg_mask.png`），修正前 = `seg_mask.png`，缺失按空 mask；
  计分用我们 W2 的 GT 与 `mask_inter_union`（同一个 IoU 定义；RefCOCO 的 GT 都非空）。
- 传给管线的模型名是我们的注册键（如 `qwen3.5-9b`），官方 `llm.py` 据名字的 `qwen` 前缀走 0–1000 网格、x 在前的约定，由它自己的代码换回像素。
