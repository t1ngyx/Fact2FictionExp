## 目标与原则
- 保留 InFact 的阶段顺序与接口：Interpretation → Pose Questions → Propose Queries → Retrieve → Answer → Judge
- 仅优化“Interpretation 与 Query Generation”两个环节，降低 LLM 调用次数与 tokens，同时保持/提升效果
- 以参数化开关实现，默认维持现状；开启后可 A/B，对比与回滚容易

## 改动范围（不变的主流程）
- 主流程保持 InFact.apply_to 不变：[infact.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/infact.py#L10-L20)
- 具体优化在 QA 基类钩子：
  - 提问生成：[_pose_questions](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L17-L23)
  - 查询生成：[propose_queries_for_question](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L45-L63)

## Interpretation 优化（问题生成）
- 缩短提示与减少输出问题数
  - 参数：n_questions（默认 10 → 可调 5–8），temperature（低温），max_tokens（限制输出长度）
  - 实施：在 PoseQuestionsPrompt 构造时注入简短模板与 n_questions；保持 interpret=True 保留原形式
- 质量控制与去重（零成本）
  - 在 _pose_questions 返回前做去重与长度阈值过滤（min_len、max_len），剔除过短/重复问题
  - 位置：[_pose_questions](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L17-L23) 增加后处理
- 批量生成与一次性解析（减少 LLM 次数）
  - 将多问题一次性生成（已有），确保模板引导返回“纯代码块列表”，降低解析成本
- 结果缓存（同文档复用）
  - 对相同 doc.id + n_questions + interpret 参数做内存缓存（一次文档多个轮次重用），减少重复生成

## Query Generation 优化（查询建议）
- 轻量启发式优先，LLM 作为回退
  - 先用启发式：
    - 将问题中专有名词和关键短语用双引号构造 1–2 条查询（关键词抽取：正则 + 简单分词）
    - 对超长问题：截取核心 n-gram（3–5 个）组合查询
  - 若检索无结果或很少结果，再调用 LLM ProposeQueries 生成增强查询
  - 参数：use_heuristic_first=True、heuristic_top_k、llm_fallback_threshold
  - 集成位置：[propose_queries_for_question](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py#L45-L63) 内部先尝试启发式
- 限制重试与查询条数
  - 限制 max_attempts（例如 1–2 次）；限制每问生成查询条数（1–3 条）
  - 参数：max_attempts、max_queries_per_question
- 轻量修饰符（可选）
  - 可开启 site:、语言限定（lang:zh）、时间范围；仅在启发式或 LLM 生成后做后处理拼接
  - 参数：enable_site_filter、site_whitelist、enable_lang_filter、enable_time_range

## 参数化与默认值（全部默认保持现状）
- 新增构造参数在 QA 基类或 InFact 变体中透出（推荐在 IInFact 中实现不变默认，按需开启）
  - interpretation：n_questions、min_len、max_len、dedup=True、temperature、max_tokens
  - query_generation：use_heuristic_first、heuristic_top_k、max_queries_per_question、max_attempts、enable_site_filter、site_whitelist、enable_lang_filter、enable_time_range
- 评测脚本透传：通过 fact_checker_kwargs / llm_kwargs 设置（参考 [evaluate.py](file:///h:/Fact2Fiction/src/scripts/averitec/evaluate.py#L14-L33)）

## 日志与度量
- 记录每文档的 LLM 调用次数与 tokens（已有 llm.get_stats）；新增：
  - 每问查询条数、启发式/LLM 命中情况、检索结果数分布
  - 提问去重前后问题数
- 位置：在 FactChecker.verify_claim 处的 meta 中追加统计项（[fact_checker.py](file:///h:/Fact2Fiction/src/infact/fact_checker.py#L125-L157)）

## 兼容性与回滚
- 默认行为不变；所有开关默认关闭或使用“等效 InFact”的参数
- 只在 iinfact 中实现这些可选开关；保留 InFact 原实现不改，便于对比与快速回滚

## 实施步骤
1. 在 iinfact 构造中加入参数但默认保持 InFact 等效
2. 在 _pose_questions 返回前增加去重与长度过滤（默认开启但保守阈值）
3. 在 propose_queries_for_question 中加入启发式优先、LLM 回退的逻辑（默认关闭）
4. 为查询条数与重试加入上限（默认与现有一致）
5. 扩展 meta 统计并在评测脚本支持参数透传
6. 用 dev 集做小样本 A/B，对比成本（LLM 调用数、tokens）与 SFR/ASR；逐项开启再叠加
