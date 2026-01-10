# iinfact 轻量化改进流程（基线一致，参数可开）

## 基线与目标
- 基线：iinfact 当前与 InFact 完全一致（纯继承，不覆写钩子）。
- 目标：在不改变默认行为的前提下，设计“轻量化、可控、可测”的改进开关；通过改参数即可启用，易做 A/B 与回滚。

## 设计原则
- 默认关闭、逐项开启；单次只开一个开关观察效果。
- 仅在变体内覆写少量钩子或插入小型策略；不影响公共管线。
- 以 LLM 低温与少重试为基调，提高稳定性与一致性。

## 改进项（参数化开关，默认关闭）
1) 查询修饰符
- 闭环：保持“问题即查询”为主，提供可选修饰符：site:、语言限定、时间范围。
- 触达位置：查询生成钩子 [base.py:propose_queries_for_question](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L45-L63)。
- 风险控制：启用时对查询长度与特殊字符做阈值过滤。

2) 专有名词引号与关键词抽取
- 策略：对专有名词保留双引号；对超长问题抽取核心关键词后再拼接查询。
- 触达位置：同上，作为轻量前处理。

3) 问题去重与长度阈值
- 策略：对生成的问题进行去重、过滤过短/过长项，降低无效问答。
- 触达位置：批量问答入口 [base.py:approach_question_batch](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L25-L43)。
- 参数：min_len、max_len、dedup=True。

4) 早停策略
- 策略：当已有 k 个一致答案或检索到高置信证据时提前停止后续问题。
- 触达位置：批量问答入口同上；通过对 q_and_a 的一致性与证据置信度统计实施。
- 参数：early_stop_k、consistency_threshold。

5) 高信誉来源优先与简易重排
- 策略：维护可信域白名单/黑名单，对搜索结果进行轻量重排，优先权威域名。
- 触达位置：资源检索与汇总 [procedure.py:retrieve_resources](file:///h:/Fact2Fiction/src/infact/procedure/procedure.py#L28-L38)，以及答案生成 [base.py:generate_answer](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L93-L105)。
- 参数：trusted_domains、blocked_domains、reorder=True。

6) 判决附加信号
- 策略：将来源数量、答案一致性、域名可信度等元信息传递给 Judge，辅助判决。
- 触达位置：判决上下文传递 [fact_checker.py:verify_claim](file:///h:/Fact2Fiction/src/infact/fact_checker.py#L125-L157)。
- 参数：use_judge_aux_signals=True。

7) 低温与少重试
- 策略：LLM temperature 保持低值，减少回答随机性；在禁用查询生成时减少重试次数。
- 触达位置：评测脚本 llm_kwargs 与 Procedure.max_attempts。
- 参数：temperature、max_attempts。

## 参数接口（建议）
- 在 IInFact 构造函数中接受可选参数，全部默认关闭或使用保守值：
  - n_questions、min_len、max_len、dedup、early_stop_k、consistency_threshold
  - trusted_domains、blocked_domains、reorder
  - query_mods（如 use_site、lang=zh、time_range）
  - use_judge_aux_signals、top_k
  - temperature、max_attempts
- 通过评测脚本传参（fact_checker_kwargs 或 llm_kwargs），保持“改参数就能跑”。
- 参考入口：[evaluate.py](file:///h:/Fact2Fiction/src/scripts/averitec/evaluate.py#L8-L33)、过程注册 [__init__.py](file:///h:/Fact2Fiction/src/infact/procedure/__init__.py#L13-L27)。

## 对应代码位置总览
- 主流程：InFact.apply_to [infact.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/infact.py#L10-L20)
- QA 基类钩子：
  - 提问 [base.py:_pose_questions](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L17-L23)
  - 批量问答与写入 [base.py:approach_question_batch](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L25-L43)
  - 查询生成 [base.py:propose_queries_for_question](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L45-L63)
  - 回答与生成 [base.py:generate_answer](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L93-L105)
- 资源检索：Procedure.retrieve_resources [procedure.py](file:///h:/Fact2Fiction/src/infact/procedure/procedure.py#L28-L38)
- 判决与统计：FactChecker.verify_claim [fact_checker.py](file:///h:/Fact2Fiction/src/infact/fact_checker.py#L125-L157)

## 验证与监控
- 指标：SFR、ASR、used_evidence 的域名分布与 Top-N 命中率、答案一致性比例。
- 策略：每次只启用一个开关，在 dev 集上做小样本（如 limit=50）A/B；效果稳定后再考虑叠加。
- 日志：在批量问答与 verify_claim 处增量打点（不改变默认行为时，先以外部统计脚本为主）。

## 风险与回滚
- 全部为“默认关闭”的增量项；开启后若效果不佳，直接关闭即可回到 InFact 等效行为。
- 不修改公共管线，变体内参数化策略可独立演化。

## 后续
- 若“问题即查询”与“去解释”组合表现持续优于基线，可在 iinfact 内提供一键开关组合消融（保持默认关闭），用于快速复现实验结论与扩大样本验证。
