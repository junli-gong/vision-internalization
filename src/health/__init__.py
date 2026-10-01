"""健康检查（简报 §8.11；DEV-0008；协议 v2：DEV-0013，D28-7；协议 v3：DEV-0016，D36-4）。

每个门在合并后的单一权重上跑一个固定的约 200 题集合（MMLU-Pro 100、GSM8K 50、IFEval 50），同时报告基座在同一集合上的
分数；任何一项下降超过 5 个点，在门报告里标红。本说明就是协议全文，在看到任何分数之前固定；数字与字符串只在
protocol.py 定义一处，tests/test_health_*.py 钉住。

协议版本 3（protocol.PROTOCOL_VERSION；D36-4 取代 D28-7 的 MMLU-Pro 标题读数；**改完即冻结为健康检查 v3，G3 之前不再动**）：
MMLU-Pro 的标题读数改为受限字母读数 mmlu_pro_letter——只看第一个生成位置（相当于只生成一个 token），限制在本题的选项字母上；
同一字母常有带空格与不带空格两种 token，取两者中较大的 logit。准确率是标题，标红只看它；同时记录正确选项的平均概率
（选项集合上的 softmax），连续读数对退化更敏感，作诊断（差值与配对 bootstrap CI，不标红）。v2 的两个读数保留为诊断、
不参与标红：256 token 的自由作答（v3 里叫 mmlu_pro_free256）与同一次生成的 32 token 前缀 mmlu_pro_fmt32。GSM8K、IFEval、
题目文件与其余协议不变。这是 MMLU-Pro 第二次改口径：v1（D26）32 token → v2（D28-7）256 token → v3（D36-4）受限字母读数。
v2 的理由仍然成立（训练会改变输出的长度与格式，自由作答的读数会把"变啰嗦了"误报成"知识退化了"），受限字母读数没有截断与
解析问题。v1、v2 的输出照旧可读、可重评、可比较（第 6 节）。

1. 题目（configs/health_check_items.json，入库；运行时不下载数据集）
   - 来源（HF 数据集，固定 commit；许可取自 HF API 的 card license）：
     - MMLU-Pro：TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be，default/test，12,032 题，MIT；
     - GSM8K：openai/gsm8k@740312add88f781978c0658806c59bc2815b9866，main/test，1,319 题，MIT；
     - IFEval：google/IFEval@966cd89545d6b6acfd7638bc708b98261ca58e84，train，541 题，Apache-2.0。
     原始文件名与 sha256 见 protocol.SOURCES；scripts/build_health_items.py 用 huggingface_hub 按 revision 下载、逐个核对，
     parquet 用 parquet_lite.py 读（不依赖 pyarrow / datasets）。
   - 抽样（种子 20260930）：排序键 sha256(f"{seed}|{benchmark}|{stage}|{id}") 的十六进制串，升序取前 k 个，即以种子为盐的
     随机置换、均匀不放回抽样，不依赖任何随机数库。
     - MMLU-Pro（id = question_id）：14 个类别各取 7 题（stage = "stratified"），再从其余题目里不分类别取 2 题
       （stage = "fill"），共 100 题；
     - GSM8K（id = test 划分里的行号，从 0 起）取 50 题；IFEval（id = key）取 50 题（stage = "uniform"）。
     每项的题目按 id 升序存放；--limit N 取每项的前 N 题。
   - 每题存原始 id 与全部评分字段：MMLU-Pro 的题干、选项、答案字母与下标，以及类别、src、抽样阶段；GSM8K 的题干与参考答案
     （参考解答末行 "#### N" 里的整数，去掉千分位逗号）；IFEval 的 prompt、instruction_id_list、kwargs（HF 文件原样）。
   - 题目文件的 sha256 由 protocol.ITEMS_SHA256 钉住，运行时核对。复现：`build_health_items.py --raw-dir <目录> --check`
     联网逐字节核对；tests 用总体索引（tests/fixtures/health/population_index.json）离线重抽、逐个比较 id。
     题目文件里记的 protocol_version 是建题时的 1（protocol.ITEMS_PROTOCOL_VERSION），v2 题目不变，文件不动。

2. 输入与生成（run.py）
   - 纯文本：adapter.build_inputs([], prompt, "instruction_first")，零张图。三个家族都支持（tests 对 200 题 × 3 个家族核对：
     没有视觉 token，指令区间解码回来就是提示）。
   - generate：enable_thinking=False；greedy，每次显式传 do_sample=False、num_beams=1、repetition_penalty=1.0、
     no_repeat_ngram_size=0（transformers 5.x 里没有显式给出的生成参数取仓库的 generation_config，Gemma 仓库默认采样）；
     max_new_tokens：MMLU-Pro 256（v1 为 32）、GSM8K 512、IFEval 1024。单轮，每题一次调用。
   - option_logits（v3，只用于 MMLU-Pro）：enable_thinking=False，选项为本题的实际字母（A 起，最多到 J）。适配器的接口约定见
     src/adapters/base.py（ModelAdapter.option_logits、OptionScores）：提示、助手头、思考开关、左填充与 generate 共用；
     一次前向，取第一个生成位置的原始 logits（HF generate 的 output_logits，logits 处理器之前，fp32；bf16 模型即 lm_head
     输出转成的 fp32）；每个字母的写法 "X" 与 " X" 由 tokenizer 查出，不是单个 token 的不用（记下来），取可用写法 logit 的
     最大值；softmax 在 fp32 下、只在本题的字母上做；argmax 的平局取 token id 小的，与 generate(allowed_token_ids=全部写法,
     max_new_tokens=1) 的 greedy 选择逐个相同（tests/test_adapter_option_logits.py 在三家族上核对）。单轮，每题一次调用。
     A–J 实际用到的写法与 token id 在预热时查一次，写进输出的 option_spellings，每题核对相同。
   - 读数（protocol.READINGS）：MMLU-Pro 每题一次 option_logits、一次生成，得到三个读数——
     - mmlu_pro_letter（标题，v3）：option_logits 的 argmax 字母；
     - mmlu_pro_free256（诊断；v2 的标题 "mmlu_pro"）：自由生成的全部，最多 256 个 token；
     - mmlu_pro_fmt32（诊断）：同一次生成的前 32 个 token（run.prefix_generation）。greedy 下生成 32 个 token 与生成 256 个
       的前 32 个逐个相同，所以只生成一次，再截出 32 的读数：token 取 token_ids 的前 32 个（适配器把 token_ids 截到第一个
       停止 token 为止，含）；raw_text、text、thinking_unclosed 由适配器的 split_answer 对这 32 个 token 解码（与适配器生成时
       拆分回答的是同一个函数，先核对它对全部 token 给出的恰是生成结果里的三个字段）；stopped 为"前 32 个 token 里出现了
       停止 token"（= 生成停在停止 token 上且生成数 ≤ 32）；n_generated_tokens 为 min(生成数, 32)；reached_max_new_tokens
       为它 ≥ 32；n_prompt_tokens 相同；wall_time_s 为 null（不单独计时）。不另调用模型，不另记账。
       tests 在三个家族的 --tiny 上核对：截出的读数与真的用 max_new_tokens=32 生成的结果逐字段相同（墙钟除外），覆盖
       停止 token 落在第 1、6、32、33、34、101 步与不停止。GPU bf16 上的核对用 --audit-fmt32 N（见第 5 节）。
     GSM8K 与 IFEval 各一个读数，名字就是项名。
   - 提示模板：
     - MMLU-Pro："{题干}\\nA. {选项 A}\\n…\\n{最后一个选项}\\nAnswer with the letter of the correct option only."
       （题干与各选项去掉首尾空白；选项字母取实际个数，最多到 J）；
     - GSM8K："{题干}\\nSolve the problem step by step. On the last line, write the final answer as '#### <number>'."；
     - IFEval：原始 prompt 去掉首尾空白（50 题里只影响 key 1999：末尾两个换行。Qwen3.5 与 Gemma 4 的模板本来就 trim，
       适配器拒绝带首尾空白的指令）；评分时官方 checker 仍拿到原始 prompt。
   - 计时前先做预热调用（generate 一次、option_logits 一次，不计入记账，D21）。记账按 src/accounting.py：每次模型调用记一次
     调用、一轮（各读数都是单轮流程），每题一个样本，样本墙钟 = 各次 build_inputs + 模型调用之和；option_logits 的计算量
     记作生成 1 个 token。MMLU-Pro 每题两次调用，GSM8K 与 IFEval 各一次。

3. 评分（scoring.py）
   - MMLU-Pro 受限字母读数（v3）：不解析文本。逐题记录的生成部分是 option_logits 的一行（选项、各选项所用 token、logit、
     选项集合上的概率、argmax 字母；run.LETTER_GENERATION_FIELDS），评分只读其中的 options、probs 与 choice
     （scoring.score_mmlu_pro_letter）：choice 等于答案字母即对；p_gold = 答案字母的概率。口径：accuracy（标题）与
     mean_p_gold（p_gold 的精确平均，百分数一位小数，另记不舍入的 mean）。没有 ParseTally。
   - MMLU-Pro 自由作答：parse_generation（先判思考段未闭合与空回答）+ src.parsing.parse_choice，letters = 本题实际的选项字母；
     解析出的字母等于答案字母即对，解析失败记错。free256 与 fmt32 用同一个评分函数，规则与 v1 的 32 token 口径完全相同。
   - 逐题记录（v2）多记 stopped（生成是否停在停止 token 上），只记录，不进入评分：与 v1 相同，评分只读 text 与
     thinking_unclosed，截断的回答照样解析（E1 的"截断判失败"不用于健康检查；改动它会改变 v1 的口径）。
   - GSM8K：两种口径都去掉千分位逗号，按精确数值与参考整数比较，相等即对（"18.00" 对 18 算对，"18.5" 算错）：
     - strict：最后一个 "####" 之后的第一个数；没有 "####" 判 no_marker，之后没有数判 no_number；
     - flexible：全文最后一个数；没有数判 no_number。
     标题口径 flexible，两种都报。数的写法见 scoring.py 的说明。
   - IFEval：官方 checker（google-research/instruction_following_eval @ e6890f85757dd84e27ca6df2dd30651dafad28e0，
     原样放在 src/health/ifeval/，只改 import 路径）的 test_instruction_following_strict / _loose；回答取剥离思考段之后的
     text。评分前 langdetect 的种子固定为 0（默认每次随机）。句子切分用 nltk ≥3.9 的 punkt_tab（english），文件哈希由
     protocol.PUNKT_TAB_ENGLISH_SHA256 钉住，真模型运行时逐个核对；数据放在 $PROJ 下，用 NLTK_DATA 指过去：
         python -m nltk.downloader -d $PROJ/nltk_data punkt_tab
         export NLTK_DATA=$PROJ/nltk_data
     标题口径 prompt-level strict；另报 instruction-level strict、prompt-level loose、instruction-level loose。
   - 分数一律是百分数，一位小数（四舍五入），全部从逐题记录重新数出来；空集合为 null。scores、parse_tally、diagnostics
     按读数给出（v3：mmlu_pro_letter、mmlu_pro_free256、mmlu_pro_fmt32、gsm8k、ifeval）；headline 每项一个数（v3 的 mmlu_pro
     为受限字母读数的准确率）。附
     ParseTally（MMLU-Pro 的两个读数，GSM8K 的 strict 与 flexible 各一份）与诊断计数（思考段未闭合、空回答、生成数达到该
     读数的 max_new_tokens，后者是截断的上界）。
   - 先生成、后评分（审查第 1 轮；punkt_tab 的许可待项目负责人答复，N12）：
     - --skip-ifeval-scoring：生成照常，IFEval 的原始输出照常保存，IFEval 各口径的 n_correct 与 pct 为 null，
       unscored 里写明原因；加载模型前不检查 punkt_tab 与 IFEval 依赖。MMLU-Pro 与 GSM8K 照常评分。
     - --rescore IN --out OUT [--only-ifeval]：不加载模型，从保存的输出重算三项（或只重算 IFEval，其余两项原样保留）。
       逐题记录分生成部分与评分部分，评分只读生成部分里的 text 与 thinking_unclosed；跑完即评与事后重评调用同一个
       函数（run.score_record），由逐题记录算出的部分（scoring.SCORED_SECTIONS）逐字节相同（tests 核对）。重评前逐题
       核对 id 与 prompt_sha256；对真模型输出重评 IFEval 同样要求钉住的 punkt_tab。
     - 每个输出文件的 scoring 段记下这次评分的时间、代码的 git commit、评了哪几项、punkt_tab 的路径与哈希、langdetect 种子；
       重评的输出还记下输入文件的路径与 sha256，输入原来的 scoring 段放进 scoring_history。

4. 比较（scoring.compare；命令行 scripts/health_check.py --compare）
   - 只比较同一题目文件、同一批题目的两次运行，协议版本须为 1、2 或 3，否则报错。比较的单位是读数；两边的版本可以不同
     （第 6 节）。
   - 每个读数每个口径给出两边的分数、差值（合并后 − 基座，百分点，按精确分数算后保留一位）、丢题数（基座对、合并后错）
     与得题数。
   - 标红只看标题读数（v3：mmlu_pro_letter 的 accuracy、gsm8k 的 flexible、ifeval 的 prompt_level_strict）：下降严格大于 5 个
     百分点即 flag = True（门报告里标红）；非标题口径只报差值（flag = None）。诊断读数 mmlu_pro_free256 与 mmlu_pro_fmt32 的
     准确率照样给出差值、CI 与符号检验，flag = None。mmlu_pro_letter 的 mean_p_gold 给出两边的平均（百分数）、按精确平均算的
     差值（百分点）与按题配对的 bootstrap 95% CI，flag = None，没有丢题 / 得题与符号检验（连续读数）。
   - 截断率提示（D28-7，与标红分开）：每个自由生成读数（受限字母读数没有截断）给出两边写满该读数 max_new_tokens 的题所占
     比例（reached_max_new_tokens）与差值，差值的绝对值严格大于 10 个百分点（protocol.TRUNCATION_NOTICE_PP）即列进 truncation.notices。IFEval 未评分时
     照样给出。
   - 与 flag 并列给出两项统计（审查第 1 轮；不改变标红规则）：
     - ci95_pp：按题配对的 bootstrap 95% CI（src.stats.paired_bootstrap_ci，percentile 法；重采样次数、置信水平与种子
       取 src.constants 的 BOOTSTRAP_N_RESAMPLES、BOOTSTRAP_CI_LEVEL、BOOTSTRAP_SEED，输出里记下实际值）。二值口径的
       统计量是均值；IFEval instruction-level 以题为重采样单位，统计量是 Σ遵循的指令数 / Σ指令数（src.stats.ratio_of_sums）。
     - sign_test：按题的精确符号检验。变好 / 变坏的题数，即该题对的单位数合并后更多 / 更少；二值口径就是得题数 / 丢题数。
       p 值为双侧二项检验（p = 1/2），按精确分数算；没有不一致的题时 p = 1。
   - 任一边某项没有评分时，该项各行 scored = False，差值、flag、CI、p 值为 null，列在 unscored 里；其余项照常比较。

5. 输出（scripts/health_check.py --out）：运行记录（src/runlog.py 的 config 与 meta：种子、git commit、题目文件 sha256、
   主机、GPU、包版本；另记 nltk / langdetect / immutabledict 的版本）、模型与适配器、生成参数与仓库 generation_config
   （v2 起另记 generation.readings：各读数的项、类型、max_new_tokens、角色与 prefix_of；v3 另记 generation.option_letters
   与 option_spellings：A–J 实际用到的写法与 token id）、逐题的原始输出与解析结果（records 按读数分键）、分数、未评分的项
   及原因、ParseTally、诊断、记账、scoring 段（见第 3 节）。
   中途出错时先把已有结果与报错写出，退出码 2。
   - --audit-fmt32 N（可选，算力机用）：跑完后 MMLU-Pro 前 N 题再真的用 max_new_tokens=32 各生成一次，按同一套规则记录与
     评分，与截出的 mmlu_pro_fmt32 逐字段比较（墙钟除外），结果写进 prefix_audit（一致题数与不一致的字段）。它核对的是
     开发机上验证不了的一件事：GPU bf16 上生成 256 的前 32 个 token 与只生成 32 个是否逐个相同。审计的调用单独记账，
     不计入 compute，不改任何记录与分数。

6. 版本与兼容（protocol.SUPPORTED_PROTOCOL_VERSIONS = (1, 2, 3)；protocol.RECORD_KEYS_BY_VERSION、HEADLINE_KEYS_BY_VERSION）
   - v1 的输出（RUN-0007、RUN-0009）：records 只有 mmlu_pro（32 token）、gsm8k、ifeval 三个键，记录里没有 stopped。
     按 v1 的布局读：--rescore 照旧（输出仍记 protocol_version 1，汇总的结构与 v1 逐字节相同），--compare 照旧。
   - v2 的输出（RUN-0011）：records 是 mmlu_pro（256 token）、mmlu_pro_fmt32、gsm8k、ifeval 四个键。按 v2 的布局读：--rescore
     照旧（输出仍记 protocol_version 2，汇总的结构与 v2 逐字节相同，tests 对 RUN-0011 的 8 份输出核对），--compare 照旧。
   - 比较时两边各按自己的版本取读数，口径（项、类型与 max_new_tokens）相同才比：v1 的 mmlu_pro 是 32 token，对上
     mmlu_pro_fmt32；v2 的 mmlu_pro 是 256 token 的自由作答，对上 mmlu_pro_free256；受限字母读数 mmlu_pro_letter 只有 v3 有。
     只要有一边没有某个读数，该读数各行 comparable = False、数值为 null，note 与 notes 写明"口径不同、标题不可比"，读数名
     列进 incomparable，不参与标红。两边都是 v1 或都是 v2 时也如此：v3 起 MMLU-Pro 的标题是受限字母读数，自由作答与 32 token
     前缀只是诊断，按 free256 / fmt32 比较、不标红（与 D31 对两边都是 v1 的裁定同一做法）。
"""
