from typing import Any, Optional
from infact.common import FCDocument, Action, Label, SearchResult
from infact.common.action import WebSearch
from infact.procedure.variants.qa_based.infact import InFact
from infact.prompts.prompt import Prompt
from infact.utils.parsing import find_code_span, extract_last_paragraph


class StrategyAndQueriesPrompt(Prompt):
    # 定义提示模板文本
    template_text = """
You are a professional fact-checker. Your task is to verify the following claim:

Claim: [CLAIM]

# Part 1: Verification Strategy
Briefly analyze the claim. Identify the key entities, specific relationships, dates, or numerical claims that must be verified. Outline a concise strategy for verification.

# Part 2: Questions and Search Queries
Based on your strategy, propose [N_QUESTIONS] pairs of specific questions and corresponding search queries.

- **Question**: A complete, self-contained natural language question that asks about a specific fact needed to verify the claim. Do not use pronouns like "he" or "it"; use full names.
- **Search Query**: A keyword-optimized query suitable for a search engine to find the answer to the question.

Please format your response exactly as follows (using backticks for the content):

Strategy: <Your strategy analysis>

1. Question: `[Question 1]`
   Query: `[Search Query 1]`

2. Question: `[Question 2]`
   Query: `[Search Query 2]`

...
"""

    def __init__(self, doc: FCDocument, n_questions: int = 10):
        # 初始化占位符，替换模板中的 [CLAIM] 和 [N_QUESTIONS]
        placeholder_targets = {
            "[CLAIM]": doc.claim,
            "[N_QUESTIONS]": n_questions
        }
        # 直接调用 super().__init__ 并传入 text 参数
        # 显式将 template_file_path 设置为 None，或者不传（默认为 None）
        # Prompt 类逻辑是：如果有 text，则直接使用；只有在没有 text 时才会去读 template_file_path
        # 但我们为了安全起见，重写 get_template 方法，或者确保父类不调用它
        super().__init__(placeholder_targets=placeholder_targets, text=self.template_text)

    # 关键修复：重写 get_template 方法
    # Prompt.compose_prompt 会调用 get_template，如果不重写，父类会尝试读取 template_file_path 导致报错
    def get_template(self) -> str:
        return self.template_text


class CollectiveAnswerPrompt(Prompt):
    template_text = """
You are a helpful assistant. Answer the following question based on the provided search results.

Question: [QUESTION]

Search Results:
[RESULTS]

Instructions:
1. Synthesize information from the search results to answer the question.
2. If the search results do not contain the answer, state "NONE".
3. If you find the answer, cite the Result ID (e.g., [Result 1]) that supports your answer.
4. Provide the final answer in a separate paragraph at the end.

Format:
Answer: <Your Answer>
Source: [Result X]
"""
    def __init__(self, question: str, results: list[SearchResult]):
        results_str = ""
        for i, res in enumerate(results):
            results_str += f"Result {i+1}:\nSource: {res.source}\nContent: {res.text}\n\n"
        
        placeholder_targets = {
            "[QUESTION]": question,
            "[RESULTS]": results_str
        }
        super().__init__(placeholder_targets=placeholder_targets, text=self.template_text)

    def get_template(self) -> str:
        return self.template_text


class IInFact(InFact):
    """
    InFact 的改进版本，结合了消融实验的发现。
    它通过添加验证策略步骤来改进“解读”阶段，并通过在生成问题时批量生成上下文相关的查询来改进“查询生成”阶段。
    这种方法显著降低了 LLM 的成本（将 N+1 次调用减少为 1 次调用）。
    """

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        # 为本次运行初始化查询缓存
        self._query_cache = {}
        # 调用父类实现来运行流程
        return super().apply_to(doc)

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        """
        在验证策略的指导下，一步生成问题和搜索查询。
        """
        # 使用自定义的提示模板
        prompt = StrategyAndQueriesPrompt(doc, n_questions=no_of_questions)
        response = self.llm.generate(prompt)

        # 提取所有的代码片段（问题和查询）
        # 预期的顺序是：问题, 查询, 问题, 查询...
        matches = find_code_span(response)
        
        questions = []
        
        # 成对迭代
        for i in range(0, len(matches) - 1, 2):
            question = matches[i]
            query = matches[i+1]
            
            questions.append(question)
            # 将该问题的查询存入缓存
            self._query_cache[question] = query
            
        return questions

    def propose_queries_for_question(self, question: str, doc: FCDocument) -> list[Action]:
        """
        从缓存中检索预生成的查询，从而节省一次 LLM 调用。
        """
        # 尝试获取缓存的查询
        if hasattr(self, '_query_cache') and question in self._query_cache:
            query_str = self._query_cache[question]
            # 确保查询被引号包裹以符合 WebSearch 的要求
            return [WebSearch(f'"{query_str}"')]
        
        # 如果未找到，回退到直接使用问题本身（类似于 NoQueryGeneration）
        return [WebSearch(f'"{question}"')]

    def generate_answer(self, question: str, results: list[SearchResult], doc: FCDocument) -> Optional[dict]:
        """
        使用“集体回答”策略：一次性阅读所有搜索结果并生成答案，
        而不是逐个结果调用 LLM。这将回答每个问题的调用次数从 M 次减少到 1 次。
        """
        prompt = CollectiveAnswerPrompt(question, results)
        # 仅调用一次 LLM
        response = self.llm.generate(prompt)

        if "NONE" in response or "None" in response:
             self.logger.debug("Got no answer.")
             return None

        # 尝试提取答案
        answer = extract_last_paragraph(response)
        
        # 简单的源提取逻辑（实际应用可能需要更复杂的正则）
        # 这里假设模型遵循了 Source: [Result X] 的格式，或者我们默认取第一个结果作为主要来源
        # 为了兼容性，我们暂时取列表中的第一个结果作为代表，或者根据模型输出解析
        # 简单起见，这里直接使用第一个结果的元数据，因为在这个粒度上很难精准匹配
        relevant_result = results[0] 

        self.logger.debug(f"Got answer: {answer}")
        qa_instance = {
            "question": question,
            "answer": answer,
            "url": relevant_result.source,
            "scraped_text": relevant_result.text
        }
        return qa_instance
