"""IFEval 官方评分代码（DEV-0008）。本文件是本项目自己写的包说明，不是上游文件。

来源：google-research/google-research 的 instruction_following_eval 目录，固定 commit UPSTREAM_COMMIT（2026-07-22，
该目录最近一次提交），许可 Apache-2.0（全文见同目录 LICENSE，取自上游仓库根目录的同一 commit）。

放进来的四个文件：instructions.py、instructions_registry.py、instructions_util.py、evaluation_lib.py。
- 原样放入，每个文件头保留上游的 Apache-2.0 声明，在声明之后加一段 "DEV-0008 vendoring note" 注释写明出处与改动；
- 只改了 import 路径（IMPORT_EDITS）：包内改成相对导入；instructions.py 的 absl.logging 换成标准库 logging，
  不新增 absl-py 依赖（该文件只调用 logging.error(fmt, *args)，两者用法相同，只影响日志输出，不影响评分）；
- evaluation_lib.py 整个文件放入，健康检查只用它的评分部分：InputExample、test_instruction_following_strict、
  test_instruction_following_loose。读写文件与打印报告的函数不用。
- tests/test_health_ifeval.py 去掉注释段、把 IMPORT_EDITS 换回原文后，逐个核对 UPSTREAM_SHA256。

运行依赖：langdetect、nltk（≥3.9.1）、immutabledict（requirements.txt）。nltk 的句子切分数据用 punkt_tab：
- 上游代码写的是 nltk.data.load("nltk:tokenizers/punkt/english.pickle")。nltk ≥ 3.9 出于安全原因不再加载 pickle，
  而是把这个路径透明地换成无 pickle 的 PunktTokenizer("english")，读 tokenizers/punkt_tab/english/ 下的表格文件；
  word_tokenize 同样用 punkt_tab。所以上游代码不用改，只需要 punkt_tab 数据。
- 数据放在项目目录下，用环境变量 NLTK_DATA 指过去，不写全局的 ~/nltk_data：
      python -m nltk.downloader -d $PROJ/nltk_data punkt_tab
      export NLTK_DATA=$PROJ/nltk_data
  文件哈希由 src.health.protocol.PUNKT_TAB_ENGLISH_SHA256 钉住（真模型运行时核对）。
"""

UPSTREAM_REPO = "https://github.com/google-research/google-research"
UPSTREAM_COMMIT = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
UPSTREAM_DIR = "instruction_following_eval"

# 上游文件（该 commit）的 sha256
UPSTREAM_SHA256 = {
    "instructions.py": "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    "instructions_registry.py": "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    "instructions_util.py": "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
    "evaluation_lib.py": "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
}
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"  # 上游仓库根目录 LICENSE

# 每个文件的改动：(上游原行, 放入后的行)，都是整行
IMPORT_EDITS = {
    "instructions.py": (
        ("from absl import logging", "import logging"),
        ("from instruction_following_eval import instructions_util", "from . import instructions_util"),
    ),
    "instructions_registry.py": (("from instruction_following_eval import instructions", "from . import instructions"),),
    "instructions_util.py": (),
    "evaluation_lib.py": (
        ("from instruction_following_eval import instructions_registry", "from . import instructions_registry"),
    ),
}
NOTE_BEGIN = "# --- DEV-0008 vendoring note (begin) ---"
NOTE_END = "# --- DEV-0008 vendoring note (end) ---"
