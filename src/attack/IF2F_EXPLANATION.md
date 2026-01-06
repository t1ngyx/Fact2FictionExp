# Fact2Fiction / IF2F 攻击方法详解

`Fact2Fiction` (在此代码库中也注册为 `if2f`) 是一种针对 Agentic Fact-Checking 系统的投毒攻击框架。它通过模拟 Fact-Checking 的过程，生成针对性极强的对抗性证据。

该框架由两个核心 Agent 组成：**Planner Agent（规划者）** 和 **Executor Agent（执行者）**。

## 1. 攻击总览

攻击的入口函数是 `create_fact2fiction_attack` (或 `create_if2f_attack`)。整体流程如下：

```python
async def create_if2f_attack(parsed_fc_report, model_name, num_fake, ...):
    
    # 1. [Planner] 分解子问题
    all_questions = await pose_questions(...)
    
    # 2. [Planner] 对抗性回答规划 (Answer Planning)
    # 核心：根据原始 Justification 生成反驳原推理逻辑的回答
    bad_qa_pairs = await infer_bad_qa_pairs(..., justification=justification)
    
    # 3. [Planner] 预算规划 (Budget Planning)
    # 核心：计算每个子问题的重要性权重，用于分配投毒预算
    question2weight = await infer_qa_weight(...)
    
    # 4. [Planner] 查询规划 (Query Planning)
    # 核心：为每个子问题生成搜索查询，用于增强检索召回率
    question2querys = await get_queries_from_questions(...)
    
    # 5. [Executor] 证据伪造 (Evidence Fabrication)
    # 核心：根据 QA 对和权重生成具体的伪造证据内容
    question2evidence = await fabricate_evidence_for_qa(..., question2weight=question2weight)
    
    # 6. [Executor] 查询拼接
    # 核心：将生成的查询拼接到证据前
    # ... (代码逻辑见下文详情)
```

---

## 2. Planner Agent：攻击规划

### 2.1 子问题分解 (Sub-question Decomposition)

攻击者首先模仿 Fact-Checker 将原 Claim 分解为多个子问题。这是由于 Agentic Fact-Checking 系统通常也是基于问题分解来验证事实的。

*   **函数**: `pose_questions`
*   **输入**: Claim
*   **输出**: List of Questions

```python
async def pose_questions(session, claim, model_name, justification=None):
    # ...
    # Prompt 模板示意：
    # "State a complete and enumerated list of 10 Questions: These are questions that probe 
    # for the veracity of the Claim..."
    # ...
```

### 2.2 回答规划 (Answer Planning) - **关键创新点**

攻击者不仅生成错误的回答，更是生成**针对性**的错误回答。

*   **函数**: `infer_bad_qa_pairs`
*   **输入**: Claim, Target Verdict (反向结论), Questions, **Original Justification**
*   **机制**: 
    *   读取受害者模型生成的原始解释（Justification）。
    *   分析原始解释中依赖的一句和推理链条。
    *   生成直接**反驳**这些推理链条的“对抗性答案 (Adversarial Answer)”。

```python
async def infer_bad_qa_pairs(..., justification=None):
    if justification:
        # Prompt 核心指令：
        # "Use this justification to identify the key reasoning patterns... 
        # then craft answers that directly contradict those patterns..."
        # 
        # 要求生成：Confident, definitive answers (自信、确定的回答)
        # 目的：不仅是错的，而且要看起来非常可信且针对性强。
```

### 2.3 预算规划 (Budget Planning)

资源（投毒数量）是有限的。为了最大化攻击效果，必须将伪造证据集中在最关键的子问题上。

*   **函数**: `infer_qa_weight`
*   **输入**: Claim, Justification, Adversarial QA Pairs
*   **机制**:
    *   让 LLM 评估每个 QA 对对于推翻原结论的“重要性分数 (Importance Score, 0-10)”。
    *   根据分数计算权重 `question2weight`。
    *   后续生成证据时，高权重的 Question 将分配到更多的伪造条目。

### 2.4 查询规划 (Query Planning)

为了让生成的伪造证据更容易被 RAG (Retrieval-Augmented Generation) 系统检索到，攻击者生成“诱饵查询”。

*   **函数**: `get_queries_from_questions`
*   **输入**: Questions
*   **机制**: 为每个子问题生成多个潜在的搜索引擎查询语句。

---

## 3. Executor Agent：攻击执行

### 3.1 证据伪造 (Evidence Fabrication)

根据规划好的 QA 对，生成具体的文档内容。

*   **函数**: `fabricate_evidence_for_qa`
*   **输入**: Bad QA Pairs, Weights
*   **机制**:
    *   依照权重采样 Question。
    *   Prompt 指令：“Craft a corpus that reflects the answer in the QA pair...”。
    *   生成的文本内容是支持那个“对抗性答案”的虚假段落。

### 3.2 查询拼接 (Concatenation)

这是提高检索率的 Trick。在将伪造证据注入知识库之前，将生成的查询拼接到证据文本的前面。

```python
# 代码逻辑片段
for item in question2evidence:
    # 找到该问题对应的查询
    queries = ... 
    
    if concat_query:
        # 随机选择一个查询拼接到证据前
        # 格式：[Search Query] + " " + [Fake Evidence Content]
        random_query = random.choice(queries)
        evidence = random_query + " " + evidence
```

**原理**: 现代检索器（Retriever）通常计算 Query 和 Document 的相似度。如果在 Document 开头直接包含可能的 Query 文本，其语义相似度或关键词匹配度会极高，从而在 Top-K 检索中排在前面，挤占真实证据的位置。

---

## 4. 消融实验参数对应

在 `if2f` 中，我们可以通过参数开关上述组件：

1.  **w/o Answer Planning**: 
    *   参数: `--no-justification`
    *   影响: `infer_bad_qa_pairs` 中的 `justification` 为 `None`，退化为生成通用的错误回答，不再针对原推理逻辑。
    
2.  **w/o Budget Planning**:
    *   参数: `--no-weighted`
    *   影响: `infer_qa_weight` 被跳过，`fabricate_evidence_for_qa` 中的 `question2weight` 为 `None`，所有问题平均分配伪造证据。

3.  **w/o Query Planning**:
    *   参数: `--no-concat-query`
    *   影响: `concat_query` 为 `False`，最后一步不进行拼接，仅使用伪造的证据文本。
