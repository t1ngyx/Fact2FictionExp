from typing import Any
from infact.common import FCDocument, Action, Label
from infact.common.action import WebSearch
from infact.procedure.variants.qa_based.infact import InFact
from infact.prompts.prompt import Prompt
from infact.utils.parsing import find_code_span


class StrategyAndQueriesPrompt(Prompt):
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

    def __init__(self, doc: FCDocument, n_questions: int = 8):
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


class IInFact(InFact):
    """
    通过添加验证策略步骤来改进“解读”阶段，并通过在生成问题时批量生成上下文相关的查询来改进“查询生成”阶段。
    这种方法显著降低了 LLM 的成本（将 N+1 次调用减少为 1 次调用）。
    """

    def apply_to(self, doc: FCDocument) -> (Label, dict[str, Any]):
        # 为本次运行初始化查询
        self._query_cache = {}
        # 调用父类实现来运行流程
        return super().apply_to(doc)

    def _pose_questions(self, no_of_questions: int, doc: FCDocument) -> list[str]:
        """
        在验证策略的指导下，一步生成问题和搜索查询。
        """
        # 使用自定义的提示模板
        prompt = StrategyAndQueriesPrompt(doc, n_questions=no_of_questions) # 不用interpretation
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
