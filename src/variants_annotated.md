# InFact Procedure Variants 源码逐行注释

## __init__.py
[__init__.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/__init__.py)

```python
# 注释：该包的初始化文件，当前为空，用于声明目录为 Python 包
```

## qa_based/__init__.py
[__init__.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/__init__.py)

```python
# 注释：qa_based 子包的初始化文件，当前为空
```

## qa_based/advanced.py
[advanced.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/advanced.py)

```python
from typing import Any, Optional
# 注释：导入类型注解 Any/Optional，描述函数返回和参数可空性

from infact.common import FCDocument, Label, SearchResult
# 注释：导入核心数据结构：事实核查文档、标签（真/假/NEI/拒答）、搜索结果

from infact.procedure.variants.qa_based.base import QABased
# 注释：导入问答式流程的基类，封装通用的提问、检索与回答逻辑

from infact.prompts.prompt import AnswerCollectively
# 注释：导入“集体回答”提示模板，用于基于一组检索结果生成综合答案

from infact.utils.parsing import extract_last_code_span, extract_last_paragraph
# 注释：导入解析工具：提取最后一个代码片段与最后一个段落（用于从 LLM 输出中抽取 ID/答案）

class AdvancedQA(QABased):
    # 注释：基于 QABased 的“高级/多迭代”版本，曾作为 InFact 的改进但在 AVeriTeC 表现较差

    """The former "dynamic" or "multi iteration" approach. Intended as improvement over
    InFact but turned out to have worse performance on AVeriTeC."""
    # 注释：类文档说明（英文），解释背景与性能结论

    def __init__(self, max_iterations: int = 3, **kwargs):
        # 注释：支持最大迭代次数配置，以及传递其他构造参数给父类
        super().__init__(**kwargs)
        # 注释：初始化父类（QABased/Procedure）依赖，例如 llm、actor、judge 等
        self.max_iterations = max_iterations
        # 注释：保存最大迭代次数，用于 apply_to 中的循环控制

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        # 注释：对输入的事实核查文档执行多轮 Q&A，直到判定不为 NEI 或达到迭代上限
        # Run iterative Q&A as long as there is NEI
        q_and_a = []
        # 注释：累积问答对的列表
        n_iterations = 0
        # 注释：迭代计数器
        label = Label.REFUSED_TO_ANSWER
        # 注释：初始标签设为拒答（在循环中更新）
        while n_iterations < self.max_iterations:
            # 注释：以最大迭代次数为上限进行循环
            n_iterations += 1
            # 注释：迭代次数自增

            questions = self._pose_questions(no_of_questions=4, doc=doc)
            # 注释：每轮生成 4 个问题（包含解读环节，由基类实现）

            new_qa_instances = self.approach_question_batch(questions, doc)
            # 注释：批量回答问题，返回问答对与检索结果列表

            q_and_a.extend(new_qa_instances)
            # 注释：累积问答对条目

            if (label := self.judge.judge(doc)) != Label.NEI:
                # 注释：使用判定器对现有推理与证据进行标签预测，若不为 NEI 则提前结束
                break

        # Fill up QA with more questions
        # 注释：若问答条目不足，补齐到 10 条（使 Q&A 更完整）
        missing_questions = 10 - len(q_and_a)
        # 注释：计算缺少的问答条数
        if missing_questions > 0:
            # 注释：如果不足则再生成若干问题
            questions = self._pose_questions(no_of_questions=missing_questions, doc=doc)
            # 注释：生成所需数量的问题
            new_qa_instances = self.approach_question_batch(questions, doc)
            # 注释：回答这些问题
            q_and_a.extend(new_qa_instances)
            # 注释：累积问答条目

        return label, dict(q_and_a=q_and_a)
        # 注释：返回最终标签与元信息（这里仅包含问答对）

    def answer_question(self,
                        question: str,
                        results: list[SearchResult],
                        doc: FCDocument = None) -> (str, SearchResult):
        # 注释：覆盖回答策略：按 5 条结果为一批，综合生成回答并选择对应的结果
        """Generates an answer to the given question by considering batches of 5 search results at once."""
        for i in range(0, len(results), 5):
            # 注释：以步长 5 遍历检索结果
            results_batch = results[i:i + 5]
            # 注释：切片取当前批次的检索结果
            prompt = AnswerCollectively(question, results_batch, doc)
            # 注释：构造“集体回答”提示，将多条证据同时喂给 LLM
            response = self.llm.generate(prompt, max_attempts=3)
            # 注释：调用 LLM 生成回答，最多尝试 3 次

            # Extract result ID and answer to the question from response
            # 注释：从 LLM 输出中提取选中的结果 ID 和最终答案
            if "NONE" not in response and "None" not in response:
                # 注释：如果不是明确的“无答案”信号，则尝试解析
                try:
                    result_id = extract_last_code_span(response)
                    # 注释：从最后一个代码片段中解析结果索引
                    if result_id != "":
                        # 注释：若非空字符串则继续
                        result_id = int(result_id)
                        # 注释：转为整数索引
                        answer = extract_last_paragraph(response)
                        # 注释：从最后一个段落中提取答案文本
                        return answer, results_batch[result_id]
                        # 注释：返回答案及被选中的检索结果
                except:
                    # 注释：解析失败时忽略并继续下一批
                    pass
        return None, None
        # 注释：若所有批次都无法得到答案，返回空
```

## qa_based/base.py
[base.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/base.py)

```python
from abc import ABC
# 注释：导入抽象基类支持，用于声明抽象流程类

from typing import Optional
# 注释：类型注解 Optional

from infact.common.action import WebSearch
# 注释：动作类型 WebSearch，表示一个带查询字符串的网页搜索动作

from infact.common import FCDocument, SearchResult
# 注释：事实核查文档结构与检索结果结构

from infact.procedure.procedure import Procedure
# 注释：通用流程基类，提供 llm/judge/actor/planner 等组件与工具方法

from infact.prompts.prompt import AnswerQuestion
# 注释：回答单个问题的提示模板（基于单条检索结果）

from infact.prompts.prompt import PoseQuestionsPrompt
# 注释：生成问题的提示模板（可包含解读阶段）

from infact.prompts.prompt import ProposeQueries, ProposeQueriesNoQuestions
# 注释：为问题生成若干搜索查询的提示模板；NoQuestions 版本在无问答模式下使用

from infact.utils.console import light_blue
# 注释：控制台输出辅助（着色）

from infact.utils.parsing import extract_last_paragraph, find_code_span, strip_string
# 注释：解析工具：提取最后段落、查找代码块（反引号包围的片段）、去除首尾空白

class QABased(Procedure, ABC):
    # 注释：问答式事实核查的抽象基类，封装通用的“提出问题→生成查询→检索→回答→判定”流程
    """Base class for all procedures that apply a questions & answers (Q&A) strategy."""

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        # 注释：调用 LLM 生成需要回答的问题列表
        """Generates some questions that needs to be answered during the fact-check."""
        prompt = PoseQuestionsPrompt(doc, n_questions=no_of_questions)
        # 注释：构造问题生成提示，包含文档与期望问题数
        response = self.llm.generate(prompt)
        # 注释：请求 LLM 输出
        # Extract the questions
        # 注释：从 LLM 输出里抽取代码片段作为问题文本（反引号内）
        questions = find_code_span(response)
        return questions
        # 注释：返回问题字符串列表

    def approach_question_batch(self, questions: list[str], doc: FCDocument) -> list:
        # 注释：批量回答问题，丢弃不可回答的问题；并记录到推理日志
        """Tries to answer the given list of questions. Unanswerable questions are dropped."""
        # Answer each question, one after another
        # 注释：逐个问题处理
        q_and_a = []
        # 注释：问答对集合
        all_search_results = []
        # 注释：收集全部检索结果（供元信息输出与后续判定）

        for question in questions:
            # 注释：遍历每个问题
            qa_instance, search_results = self.approach_question(question, doc)
            # 注释：对单个问题进行检索与回答，返回问答条目与检索结果列表
            if qa_instance is not None:
                # 注释：若得到答案则记录
                q_and_a.append(qa_instance)
                all_search_results.extend(search_results)
                # 注释：累积检索结果

        # Add Q&A to doc reasoning
        # 注释：将问答内容以 Markdown 形式追加到文档推理区
        q_and_a_strings = [(f"### {triplet['question']}\n"
                            f"Answer: {triplet['answer']}\n\n"
                            f"Source URL: {triplet['url']}") for triplet in q_and_a]
        # 注释：为每个问答构造展示字符串（问题、答案、来源 URL）
        q_and_a_string = "## Initial Q&A\n" + "\n\n".join(q_and_a_strings)
        # 注释：组合为整体块，标题“Initial Q&A”
        doc.add_reasoning(q_and_a_string)
        # 注释：写入推理日志

        return q_and_a, all_search_results
        # 注释：返回问答集合与检索结果集合

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[WebSearch]:
        # 注释：为单个问题生成若干搜索查询（WebSearch 动作列表）
        prompt = ProposeQueries(question, doc)
        # 注释：构造查询生成提示（包含问题与文档上下文）

        n_tries = 0
        # 注释：尝试计数
        while n_tries < self.max_attempts:
            # 注释：循环重试直至达到最大尝试次数
            n_tries += 1
            response = self.llm.generate(prompt)
            # 注释：向 LLM 请求查询建议
            if response is None:
                # 注释：若无输出则警告并重试
                self.logger.warning("WARNING: No new actions were found. Retrying...")
                continue
            queries = extract_queries(response)
            # 注释：解析反引号代码块，转换为 WebSearch 动作列表

            if len(queries) > 0:
                # 注释：若有查询则返回
                return queries

            self.logger.info("WARNING: No new actions were found. Retrying...") 
            # 注释：记录无查询的提示信息
        
        # Return empty list if no queries found after max attempts
        # 注释：超过最大尝试次数仍无查询时返回空列表
        return []

    def approach_question(self, question: str, doc: FCDocument = None) -> Optional[tuple]:
        # 注释：针对单个问题执行完整流程：生成查询→检索→生成答案
        """Tries to answer the given question. If unanswerable, returns (None, [])."""
        self.logger.debug(light_blue(f"Answering question: {question}"))
        # 注释：调试日志标记当前处理的问题
        self.actor.reset()
        # 注释：重置执行器状态，避免上一轮残留

        # Stage 3: Generate search queries
        # 注释：阶段 3：生成搜索查询
        queries = self.propose_queries_for_question(question, doc)
        # 注释：调用上面的方法生成 WebSearch 列表
        if queries is None or len(queries) == 0:
            # 注释：无查询则无法继续，返回无答案与空检索结果
            return None, []

        # Execute searches and gather all results
        # 注释：执行检索动作，聚合检索结果
        search_results = self.retrieve_resources(queries)
        # 注释：调用基类/执行器进行实际网络检索

        # Step 4: Answer generation
        # 注释：阶段 4：答案生成
        if len(search_results) > 0:
            # 注释：有检索结果则尝试生成答案
            return self.generate_answer(question, search_results, doc), search_results
        else:
            # Return (None, []) when no search results to maintain tuple structure
            # 注释：无检索结果时返回空，保持返回结构一致
            return None, []

    def answer_question(self,
                        question: str,
                        results: list[SearchResult],
                        doc: FCDocument = None) -> (str, SearchResult):
        # 注释：默认回答策略：调用“逐个结果尝试”的方法，返回答案与相关结果
        """Answers the given question and returns the answer along with the ID of the most relevant result."""
        answer, relevant_result = self.answer_question_individually(question, results, doc)
        # 注释：逐条尝试，从第一个能给出答案的结果返回
        return answer, relevant_result

    def generate_answer(self, question: str, results: list[SearchResult], doc: FCDocument) -> Optional[dict]:
        # 注释：将“答案 + 元数据”组装为问答条目（包含来源 URL 与抓取文本）
        answer, relevant_result = self.answer_question(question, results, doc)
        # 注释：先获取答案与选中的结果
        if answer is not None:
            # 注释：若存在答案则记录并构造问答条目
            self.logger.debug(f"Got answer: {answer}")
            qa_instance = {"question": question,
                           "answer": answer,
                           "url": relevant_result.source,
                           "scraped_text": relevant_result.text}
            # 注释：包含问题、答案、来源 URL 与正文文本
            return qa_instance
        else:
            # 注释：没有答案则输出调试信息
            self.logger.debug("Got no answer.")

    def answer_question_individually(
            self,
            question: str,
            results: list[SearchResult],
            doc: FCDocument
    ) -> (Optional[str], Optional[SearchResult]):
        # 注释：逐条检索结果尝试生成答案；第一个成功的结果即为相关结果
        """Generates an answer to the given question by iterating over the search results
        and using them individually to answer the question."""
        for result in results:
            # 注释：遍历检索结果
            answer = self.attempt_answer_question(question, result, doc)
            # 注释：基于单条结果构造提示并尝试回答
            if answer is not None:
                # 注释：若成功得到答案则返回答案与该结果
                return answer, result
        return None, None
        # 注释：所有结果均失败时返回空

    def attempt_answer_question(self, question: str, result: SearchResult, doc: FCDocument) -> Optional[str]:
        # 注释：针对单条检索结果生成回答
        """Generates an answer to the given question."""
        prompt = AnswerQuestion(question, result, doc)
        # 注释：构造“基于单条结果回答”的提示
        response = self.llm.generate(prompt, max_attempts=3)
        # 注释：向 LLM 请求回答，最多尝试 3 次
        # Extract answer from response
        # 注释：从 LLM 输出中抽取答案段落
        if "NONE" not in response and "None" not in response:
            # 注释：过滤“无答案”标记
            try:
                answer = extract_last_paragraph(response)
                # 注释：提取最后一个段落作为答案
                return answer
            except:
                # 注释：解析失败则忽略
                pass

def extract_queries(response: str) -> list[WebSearch]:
    # 注释：工具函数：从 LLM 响应中提取反引号包围的查询片段，并转为 WebSearch 动作
    matches = find_code_span(response)
    # 注释：找到所有代码片段（通常是查询字符串）
    actions = []
    # 注释：累积动作列表
    for match in matches:
        # 注释：遍历每个片段
        query = strip_string(match)
        # 注释：去除空白，得到实际查询字符串
        action = WebSearch(f'"{query}"')
        # 注释：将查询用双引号包裹，构造 WebSearch 动作
        actions.append(action)
    return actions
    # 注释：返回动作列表
```

## qa_based/first_result.py
[first_result.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/first_result.py)

```python
from infact.common import FCDocument, SearchResult
# 注释：导入文档与检索结果类型

from infact.procedure.variants.qa_based.infact import InFact
# 注释：继承标准 InFact 流程（问答式）

class FirstResult(InFact):
    # 注释：变体：总是使用第一个检索结果进行回答
    """InFact but using always the first result."""

    def answer_question(self,
                        question: str,
                        results: list[SearchResult],
                        doc: FCDocument = None) -> (str, SearchResult):
        # 注释：覆盖回答策略：直接取第一个结果，不做遍历挑选
        relevant_result = results[0]
        # 注释：选定第一个检索结果
        answer = self.attempt_answer_question(question, relevant_result, doc)
        # 注释：基于该结果生成答案
        return answer, relevant_result
        # 注释：返回答案与所用结果
```

## qa_based/iinfact.py
[iinfact.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/iinfact.py)

```python
from typing import Any
# 注释：导入 Any 类型注解

from infact.common import FCDocument, Action, Label
# 注释：导入文档、通用动作类型、标签类型

from infact.common.action import WebSearch
# 注释：导入 WebSearch 动作构造器

from infact.procedure.variants.qa_based.infact import InFact
# 注释：继承标准 InFact 流程

from infact.prompts.prompt import Prompt
# 注释：导入通用 Prompt 基类

from infact.utils.parsing import find_code_span
# 注释：导入解析工具：提取反引号包围的代码片段

class StrategyAndQueriesPrompt(Prompt):
    # 注释：自定义 Prompt：一并生成“验证策略 + 问题-查询配对”
    # 定义提示模板文本
    template_text = """# Instructions
You are a fact-checker. Your overall motivation is to verify a given Claim. You are at the beginning of the fact-check, i.e. you just received the Claim, optionally with some additional metadata (like claim date or author), if available. **Your task right now is to prepare the fact-check.** That is,

1. **Verification Strategy**: Briefly analyze the claim. Identify the key entities, specific relationships, dates, or numerical claims that must be verified. Outline a concise strategy for verification.
2. **Questions and Search Queries**: Based on your strategy, propose [N_QUESTIONS] pairs of specific questions and corresponding search queries.

IMPORTANT: Follow these rules:
* State every single question in a way that it can be understood independently and without additional context. Therefore, be explicit and do not use pronouns or generic terms in place of names or objects.
* Enclose each single question with backticks like `this`.
* Enclose each single search query with backticks like `this`.
* The Search Query should be keyword-optimized suitable for a search engine to find the answer to the question.

# Examples
Claim: "New Zealand’s new Food Bill bans gardening"
Strategy: The claim is about a specific bill in New Zealand. I need to check if such a bill exists and if it contains provisions banning gardening.
1. Question: `Did New Zealand's government pass a food bill that restricted gardening activities for its citizen?`
   Query: `New Zealand Food Bill gardening ban`

2. Question: `What are the provisions of New Zealand's Food Bill regarding home gardening?`
   Query: `New Zealand Food Bill home gardening provisions`

# The Claim
[CLAIM]

# Verification Strategy
"""
    # 注释：多行字符串模板，要求生成策略以及成对问题+查询，问题与查询均以反引号包住；包含示例与占位符

    def __init__(self, doc: FCDocument, n_questions: int = 8):
        # 注释：构造时提供文档与问题数，准备占位符替换
        # 初始化占位符，替换模板中的 [CLAIM] 和 [N_QUESTIONS]
        placeholder_targets = {
            "[CLAIM]": doc.claim,
            # 注释：将文档中的 claim 文本替换到模板
            "[N_QUESTIONS]": n_questions
            # 注释：将问题数替换到模板
        }
        # 直接调用 super().__init__ 并传入 text 参数
        # 显式将 template_file_path 设置为 None，或者不传（默认为 None）
        # Prompt 类逻辑是：如果有 text，则直接使用；只有在没有 text 时才会去读 template_file_path
        # 但我们为了安全起见，重写 get_template 方法，或者确保父类不调用它
        super().__init__(placeholder_targets=placeholder_targets, text=self.template_text)
        # 注释：传入模板文本与占位符给基类，避免读取外部文件

    # 关键修复：重写 get_template 方法
    # Prompt.compose_prompt 会调用 get_template，如果不重写，父类会尝试读取 template_file_path 导致报错
    def get_template(self) -> str:
        # 注释：返回内置的模板文本，确保 compose_prompt 能正常工作
        return self.template_text

class IInFact(InFact):
    # 注释：改进版 InFact：在“解读/策略”阶段一次生成问题与查询，降低 LLM 调用次数
    """
    通过添加验证策略步骤来改进“解读”阶段，并通过在生成问题时批量生成上下文相关的查询来改进“查询生成”阶段。
    这种方法显著降低了 LLM 的成本（将 N+1 次调用减少为 1 次调用）。
    """

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        # 注释：重置本次运行的查询缓存，然后调用父类流程
        # 为本次运行初始化查询
        self._query_cache = {}
        # 注释：用于保存每个问题对应的预生成查询
        # 调用父类实现来运行流程
        return super().apply_to(doc)
        # 注释：沿用 InFact 的整体执行逻辑

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        # 注释：覆盖提问阶段：一次性生成“问题 + 查询”，并将查询缓存起来
        """
        在验证策略的指导下，一步生成问题和搜索查询。
        """
        # 使用自定义的提示模板
        prompt = StrategyAndQueriesPrompt(doc, n_questions=no_of_questions) # 不用interpretation
        # 注释：构造策略驱动的模板，不启用单独的 interpret 标志
        response = self.llm.generate(prompt)
        # 注释：调用 LLM 生成文本

        # 提取所有的代码片段（问题和查询）
        # 预期的顺序是：问题, 查询, 问题, 查询...
        matches = find_code_span(response)
        # 注释：将反引号包住的片段按顺序取出

        questions = []
        # 注释：预备问题列表
        
        # 成对迭代
        for i in range(0, len(matches) - 1, 2):
            # 注释：每次取两个片段：第一个为问题，第二个为查询
            question = matches[i]
            query = matches[i+1]
            # 注释：成对读取

            questions.append(question)
            # 注释：记录问题
            # 将该问题的查询存入缓存
            self._query_cache[question] = query
            # 注释：缓存对应的查询，后续直接使用而无需再调用 LLM
            
        return questions
        # 注释：返回问题列表

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[Action]:
        # 注释：覆盖查询生成：优先从缓存中取预生成的查询
        """
        从缓存中检索预生成的查询，从而节省一次 LLM 调用。
        """
        # 尝试获取缓存的查询
        if hasattr(self, '_query_cache') and question in self._query_cache:
            # 注释：存在缓存则取出并包装为 WebSearch 动作
            query_str = self._query_cache[question]
            # 注释：取出对应查询
            # 确保查询被引号包裹以符合 WebSearch 的要求
            return [WebSearch(f'"{query_str}"')]
        
        # 如果未找到，回退到直接使用问题本身（类似于 NoQueryGeneration）
        return [WebSearch(f'"{question}"')]
        # 注释：无缓存时将问题本身作为查询
```

## qa_based/infact.py
[infact.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/infact.py)

```python
from infact.common import FCDocument, Label
# 注释：导入事实核查文档与标签类型

from infact.procedure.variants.qa_based.base import QABased
# 注释：基于问答的流程基类

from typing import Any
# 注释：类型注解 Any

class InFact(QABased):
    # 注释：标准 InFact 流程实现：包含六阶段（第六阶段的“论证生成”在方法外进行）
    """The procedure as implemented by InFact, using all six stages (stage 6, justification
    generation, follows outside of this method)."""

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        # 注释：执行完整流程：提问→检索与回答→判定→返回结果与元信息
        # Stage 1 & 2: Interpretation & Question posing
        # 注释：阶段 1&2：解读与提出问题
        questions = self._pose_questions(no_of_questions=10, doc=doc)
        # 注释：生成 10 个问题

        # Stages 3 & 4: Search query generation and question answering
        # 注释：阶段 3&4：生成查询并回答问题
        q_and_a, search_results = self.approach_question_batch(questions, doc)
        # 注释：批量处理所有问题，得到问答对与检索结果集合

        # Stage 5: Veracity prediction
        # 注释：阶段 5：基于已有推理与证据进行真伪判定
        label = self.judge.judge(doc)
        # 注释：得到标签（如 SUPPORTS/REFUTES/NEI/拒答）

        return label, dict(q_and_a=q_and_a, used_evidence=search_results)
        # 注释：返回标签以及元信息：问答内容与使用过的证据（检索结果）
```

## qa_based/naive.py
[naive.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/naive.py)

```python
from typing import Any
# 注释：类型注解 Any

from infact.common import FCDocument, Label
# 注释：导入文档与标签类型

from infact.procedure.variants.qa_based.base import QABased
# 注释：继承问答基类

class NaiveQA(QABased):
    # 注释：最朴素的策略：直接让判定器对声明做“朴素判定”，不进行检索/回答
    """The naivest-possible approach where the claim veracity is
    predicted right away from the claim."""
    # 注释：类文档说明（英文）

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        # 注释：直接使用 judge.judge_naively，对声明文本进行简单判定
        verdict = self.judge.judge_naively(doc)
        # 注释：得到初步结论标签
        meta = dict(q_and_a=[])
        # 注释：元信息中记录空的问答集合
        return verdict, meta
        # 注释：返回标签与元信息
```

## qa_based/no_evidence.py
[no_evidence.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/no_evidence.py)

```python
from infact.common import FCDocument
# 注释：导入文档类型

from infact.procedure.variants.qa_based.infact import InFact
# 注释：继承标准 InFact 流程

from infact.prompts.prompt import AnswerQuestionNoEvidence
# 注释：导入“无证据回答”提示模板（不进行检索，直接回答）

class NoEvidence(InFact):
    # 注释：变体：不做证据检索，直接对问题作答
    """InFact but without any evidence retrieval."""

    def approach_question_batch(self, questions: list[str], doc: FCDocument) -> list:
        # 注释：覆盖批量回答：不检索，直接让 LLM 回答，并写入推理日志
        q_and_a = []
        # 注释：问答条目集合
        doc.add_reasoning("## Research Q&A")
        # 注释：在推理区域添加标题
        for question in questions:
            # 注释：遍历每个问题
            prompt = AnswerQuestionNoEvidence(question, doc)
            # 注释：构造“无证据回答”的提示
            response = self.llm.generate(prompt)
            # 注释：请求 LLM 输出回答
            qa_string = (f"### {question}\n"
                         f"Answer: {response}")
            # 注释：构造展示字符串：问题与答案
            doc.add_reasoning(qa_string)
            # 注释：写入推理日志
            qa_instance = {
                "question": question,
                "answer": response,
                "url": "",
                "scraped_text": "",
            }
            # 注释：问答条目的结构，无来源和正文文本
            q_and_a.append(qa_instance)
            # 注释：加入集合
        return q_and_a, []
        # 注释：返回问答集合与空的检索结果集合
```

## qa_based/no_interpretation.py
[no_interpretation.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/no_interpretation.py)

```python
from infact.procedure.variants.qa_based.infact import InFact
# 注释：继承标准 InFact

from infact.common.document import FCDocument
# 注释：从 common.document 路径导入文档类型（同前面的 common 模块不同文件）

from infact.utils.parsing import find_code_span
# 注释：解析工具：提取代码片段（反引号内文本）

from infact.prompts.prompt import PoseQuestionsPrompt
# 注释：用于生成问题的提示模板

class NoInterpretation(InFact):
    # 注释：变体：不做“解读”阶段（interpretation），直接生成问题
    """InFact but without interpretation."""

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        # 注释：覆盖提问方法：将 interpret=False 传给提示以跳过解读
        """Generates some questions that needs to be answered during the fact-check."""
        prompt = PoseQuestionsPrompt(doc, n_questions=no_of_questions, interpret=False)
        # 注释：构造提示，显式禁用解读
        response = self.llm.generate(prompt)
        # 注释：调用 LLM 生成输出
        # Extract the questions
        # 注释：提取反引号包裹的问题列表
        questions = find_code_span(response)
        return questions
        # 注释：返回问题集合
```

## qa_based/no_query_gen.py
[no_query_gen.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/no_query_gen.py)

```python
from infact.common import FCDocument, Action
# 注释：导入文档与通用动作类型

from infact.common.action import WebSearch
# 注释：导入 WebSearch 动作

from infact.procedure.variants.qa_based.infact import InFact
# 注释：继承标准 InFact

class NoQueryGeneration(InFact):
    # 注释：变体：直接用问题文本当作搜索查询（不再生成额外查询）
    """InFact but using the questions as search queries directly (instead of generating some)."""

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[Action]:
        # 注释：将问题本身包裹引号，作为 WebSearch 的查询字符串
        return [WebSearch(f'"{question}"')]
        # 注释：返回单个 WebSearch 动作
```

## qa_based/simple.py
[simple.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/qa_based/simple.py)

```python
from typing import Optional
# 注释：导入 Optional 类型注解

from infact.common import FCDocument, SearchResult, Action
# 注释：导入文档、检索结果与动作类型

from infact.procedure.variants.qa_based.infact import InFact
# 注释：继承标准 InFact

from infact.prompts.prompt import ProposeQuerySimple
# 注释：“简单查询”提示模板：为问题提一个查询

from infact.procedure.variants.qa_based.base import extract_queries
# 注释：从基类导入查询解析工具

class SimpleQA(InFact):
    # 注释：简化版流程：不做解读，仅用一个查询，并选用第一个检索结果
    """InFact but without interpretation, uses only one query per question and takes first search result.
    (Never used in AVeriTeC challenge)."""

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[Action]:
        # 注释：为单个问题生成一个简单查询，若失败则重试，最终可能返回空
        prompt = ProposeQuerySimple(question)
        # 注释：构造简单查询提示

        n_tries = 0
        # 注释：重试计数
        while n_tries < self.max_attempts:
            # 注释：在最大次数内重试
            n_tries += 1
            response = self.llm.generate(prompt)
            # 注释：请求 LLM 生成查询
            queries = extract_queries(response)
            # 注释：从响应中提取反引号包住的查询

            if len(queries) > 0:
                # 注释：若成功得到至少一个查询，返回其第一个
                return [queries[0]]

            self.logger.info("No new actions were found. Retrying...")
            # 注释：记录未找到查询的信息

        self.logger.warning("Got no search query, dropping this question.")
        # 注释：超过最大次数仍没有查询，警告并放弃该问题
        return []
        # 注释：返回空动作列表

    def answer_question(self,
                        question: str,
                        results: list[SearchResult],
                        doc: FCDocument = None) -> (str, SearchResult):
        # 注释：回答策略：直接使用第一个检索结果
        relevant_result = results[0]
        # 注释：选第一条结果作为证据
        answer = self.attempt_answer_question(question, relevant_result, doc)
        # 注释：基于该结果生成答案
        return answer, relevant_result
        # 注释：返回答案与所用结果
```

## summary_based/__init__.py
[__init__.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/summary_based/__init__.py)

```python
# 注释：summary_based 子包的初始化文件，当前为空
```

## summary_based/default.py
[default.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/summary_based/default.py)

```python
from typing import Any, Collection
# 注释：导入类型 Any，集合类型 Collection

from infact.common import FCDocument, Label, Evidence
# 注释：导入文档、标签与证据类型（Evidence 对应检索/提取到的事实依据）

from infact.procedure.procedure import Procedure
# 注释：继承通用流程基类

from infact.prompts.prompt import ReiteratePrompt
# 注释：导入“重述/巩固”提示，用于整合知识与提出新问题

class DynamicSummary(Procedure):
    # 注释：基于“总结/巩固”的动态多迭代流程（非问答式）
    def __init__(self, max_iterations: int = 3, **kwargs):
        # 注释：支持最大迭代次数与父类参数传递
        super().__init__(**kwargs)
        # 注释：初始化通用组件
        self.max_iterations = max_iterations
        # 注释：保存迭代上限

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        # 注释：循环执行：规划动作→执行→总结→判定，直到不再 NEI 或达迭代上限
        all_evidences = []
        # 注释：累积全部证据
        n_iterations = 0
        # 注释：迭代计数
        label = Label.NEI
        # 注释：初始标签为信息不足（NEI）
        while label == Label.NEI and n_iterations < self.max_iterations:
            # 注释：只要仍信息不足且未超上限就继续
            self.logger.info("Not enough information yet. Continuing fact-check...")
            # 注释：日志：信息不足，继续事实核查
            n_iterations += 1
            # 注释：迭代自增
            actions, reasoning = self.planner.plan_next_actions(doc)
            # 注释：通过规划器得到下一步动作以及推理文本（可能为空）
            if len(reasoning) > 32:  # Only keep substantial reasoning
                # 注释：仅当推理文本较有信息量时才写入文档（降噪）
                doc.add_reasoning(reasoning)
            if actions:
                # 注释：如果有动作，则追加到文档的动作列表
                doc.add_actions(actions)
            else:
                # 注释：若无法规划出有用动作则停止循环
                break  # the planner wasn't able to determine further useful actions, giving up
            evidences = self.actor.perform(actions, doc)
            # 注释：由执行器执行动作并收集证据
            all_evidences.extend(evidences)
            # 注释：累积证据
            doc.add_evidence(evidences)  # even if no evidence, add empty evidence block for the record
            # 注释：将证据写入文档，即便为空也添加记录
            self._consolidate_knowledge(doc, evidences)
            # 注释：调用巩固知识，将当前证据整合为新的推理/问题
            label = self.judge.judge(doc)
            # 注释：判定当前状态的真伪结论
        return label, dict(used_evidence=all_evidences)
        # 注释：返回最终标签以及所有使用过的证据

    def _consolidate_knowledge(self, doc: FCDocument, evidences: Collection[Evidence]):
        # 注释：将当前证据与上下文总结/重述，产生新的推理与可能的问题
        """Analyzes the currently available information and states new questions, adds them
        to the FCDoc."""
        prompt = ReiteratePrompt(doc, evidences)
        # 注释：构造“重述”提示，输入为文档+证据集合
        answer = self.llm.generate(prompt)
        # 注释：请求 LLM 输出整合后的总结/新问题
        doc.add_reasoning(answer)
        # 注释：将该总结写回文档
```

## summary_based/no_qa.py
[no_qa.py](file:///h:/Fact2Fiction/src/infact/procedure/variants/summary_based/no_qa.py)

```python
from typing import Any
# 注释：导入 Any 类型

from infact.common import FCDocument, Label
# 注释：导入文档与标签类型

from infact.common.action import WebSearch
# 注释：导入 WebSearch 动作构造器

from infact.procedure.procedure import Procedure
# 注释：继承通用流程基类

from infact.procedure.variants.qa_based.base import extract_queries
# 注释：复用查询解析工具

from infact.prompts.prompt import ProposeQueriesNoQuestions
# 注释：用于“无问答”的查询生成提示

class StaticSummary(Procedure):
    # 注释：静态总结流程：不提出问题，直接生成查询→检索→摘要→判定
    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        """InFact but omitting posing any questions."""
        # Stage 2*: Search query generation (modified)
        # 注释：阶段 2*：直接为声明生成搜索查询（不走问答）
        queries = self.generate_search_queries(doc)
        # 注释：生成查询列表

        # Stage 3*: Evidence retrieval (modified)
        # 注释：阶段 3*：执行检索并做摘要化（summarize=True）
        results = self.retrieve_resources(queries, summarize=True, doc=doc)
        # 注释：调用执行器检索，并请求对结果进行摘要
        doc.add_reasoning("## Web Search")
        # 注释：推理区添加标题
        used_evidence = []
        # 注释：收集有用的证据
        for result in results[:10]:
            # 注释：仅查看前 10 条结果
            if result.is_useful():
                # 注释：过滤有用结果
                used_evidence.append(result)
                # 注释：加入证据集合
                summary_str = f"### Search Result\n{result}"
                # 注释：构造结果摘要字符串（result 的 __str__ 提供格式）
                doc.add_reasoning(summary_str)
                # 注释：写入推理区
        # Stage 4: Veracity prediction
        # 注释：阶段 4：判定真伪
        label = self.judge.judge(doc)
        # 注释：得到标签
        return label, {"used_evidence": used_evidence}
        # 注释：返回标签与被使用的证据列表

    def generate_search_queries(self, doc: FCDocument) -> list[WebSearch]:
        # 注释：生成查询的循环逻辑：直到得到查询或达到最大尝试次数
        prompt = ProposeQueriesNoQuestions(doc)
        # 注释：构造“无问答”查询生成提示

        n_tries = 0
        # 注释：重试计数
        while True:
            # 注释：开放式循环，由内部条件控制退出
            n_tries += 1
            response = self.llm.generate(prompt)
            # 注释：调用 LLM 生成查询
            queries = extract_queries(response)
            # 注释：解析反引号包住的查询

            if len(queries) > 0 or n_tries == self.max_attempts:
                # 注释：如果已经得到查询或达到最大重试，则返回
                return queries

            self.logger.info("WARNING: No new actions were found. Retrying...")
            # 注释：否则记录警告并继续下一轮
```
