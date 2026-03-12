# Fact2Fiction / IF2F 攻击方法详解（代码级解释）

本文面向代码阅读与实验复现，按代码流程解释 `Fact2Fiction` 与 `IF2F` 的实现。所有函数均位于 `src/attack/attack_methods.py`，入口由 `src/attack/main.py` 调用。

---

## 1. Fact2Fiction：主流程与关键函数

入口函数：`create_fact2fiction_attack`

### 1.1 Planner Agent（规划阶段）

1) **子问题分解**：`pose_questions`  
输入：`claim`  
输出：`List[str]`（最多 10 个问题）  
作用：模拟事实核查系统的分解策略，为后续攻击生成“问题骨架”。

2) **回答规划**：`infer_bad_qa_pairs`  
输入：`claim` + `target_verdict` + `questions` + `justification`  
输出：`[{question, answer, reason}]`  
作用：生成“对抗性答案”，并利用原始 justification 的推理链进行反向构造。

3) **预算规划**：`infer_qa_weight`  
输入：`bad_qa_pairs` + `justification`  
输出：`Dict[question -> weight]`  
作用：让 LLM 给每个问题打重要性分数，并归一化为权重，用于分配伪造证据数量。

4) **查询规划**：`get_queries_from_questions`  
输入：`questions`  
输出：`[{question, queries}]`  
作用：为检索阶段生成“诱饵查询”，用于增强伪造证据被检索到的概率。

### 1.2 Executor Agent（执行阶段）

5) **证据伪造**：`fabricate_evidence_for_qa`  
输入：`bad_qa_pairs` + `question2weight`  
输出：按问题生成多条伪造证据  
作用：对高权重问题生成更多伪造证据，强化攻击重点。

6) **查询拼接**：在 `create_fact2fiction_attack` 内完成  
做法：将 query 拼接到 evidence 前（`query + " " + evidence`）  
目的：提高语义检索命中率，挤占真实证据的 Top‑K 位置。

---

## 2. IF2F：在原结构上做低成本改进

入口函数：`create_if2f_attack`  

IF2F 保留 Fact2Fiction 的 Planner/Executor 结构，仅在**规划阶段中间插入轻量剪枝**，以降低 API 调用并提升证据相关性。同时在查询拼接阶段使用**上下文相关性优选 query**并拼接 `claim_text`，增强检索一致性。

### 2.1 IF2F 与 Fact2Fiction 的差异

相同点：
- 同样包含 **分解问题 → 回答规划 → 预算规划 → 查询规划 → 证据生成 → 查询拼接** 的完整链路。

不同点：
- 在 **Answer Planning 之后**新增 **问题剪枝**，只保留相关性高的问题进入后续规划与生成。
- 在 **Query 拼接阶段**选择与上下文最相关的 query，并拼接 `claim_text`。

### 2.2 剪枝触发位置

剪枝发生在 `create_if2f_attack` 的 Step 2.5：
- 发生时间：`infer_bad_qa_pairs` 之后
- 影响范围：`infer_qa_weight`、`get_queries_from_questions`、`fabricate_evidence_for_qa` 的输入问题集合
- 目标：减少低相关问题造成的预算分散与 API 调用浪费

---

## 3. IF2F 相关性判断（BM25 + 实体重叠）

剪枝使用**混合评分**：
- **BM25 相似度**：衡量问题文本与 `claim + justification` 的语义相关性
- **实体重叠率**：衡量问题中的关键实体与上下文一致性

最终分数：
```
hybrid_score = 0.7 * bm25_score + 0.3 * entity_overlap
```

### 3.1 BM25 计算

在代码中使用轻量 BM25 实现（不引入外部库），以 `context_text` 为 query，以问题集合为 docs：

```python
def bm25_scores(query_text, docs_tokens, k1=1.5, b=0.75):
    # 计算每个问题的 BM25 分数
```

### 3.2 实体识别（规则扩展）

claim 为英文时，使用规则抽取实体：

- **专有名词短语**（Title Case）
- **缩写**（全大写）
- **数字**
- **引号内片段**

```python
def extract_entities(text):
    entities = set()
    entities.update(re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text))
    entities.update(re.findall(r"\b[A-Z]{2,}\b", text))
    entities.update(re.findall(r"\b\d{2,}\b", text))
    entities.update(re.findall(r"\"([^\"]{3,})\"", text))
    return set(e.strip() for e in entities if e.strip())
```

### 3.3 剪枝策略

默认保留 4~8 个高分问题，具体数量由 `num_fake` 控制：

```python
max_questions = min(len(bad_qa_pairs), max(4, min(8, num_fake)))
```

---

## 4. 参数与运行方式

### 4.1 关键参数

- `--no-if2f-prune`  
  关闭剪枝，恢复原始问题数量

- `--if2f-prune-method token`  
  使用上一版本的 token overlap 剪枝方式

- `--if2f-bm25-weight`  
  设置 BM25 在混合评分中的权重（默认 0.7）

- `--no-concat-query`  
  关闭查询拼接（不影响剪枝）

---

## 4. IF2F 查询拼接优化

IF2F 在拼接阶段选择最相关的 query，并强制拼接 `claim_text`：

```
evidence = best_query + " " + claim_text + " " + evidence
```

`best_query` 通过与 `claim_text + justification` 的词项重叠比例进行排序选取（无额外 API 成本）。

### 4.2 运行示例

```bash
# 默认：混合评分剪枝
python -m attack.main --attack-type if2f ...

# 上一版本剪枝（token overlap）
python -m attack.main --attack-type if2f --if2f-prune-method token

# 关闭剪枝
python -m attack.main --attack-type if2f --no-if2f-prune
```

---

## 5. 代码结构索引

- `create_fact2fiction_attack`：Fact2Fiction 主流程  
- `create_if2f_attack`：IF2F 主流程（含剪枝）
- `pose_questions`：问题分解  
- `infer_bad_qa_pairs`：回答规划  
- `infer_qa_weight`：预算规划  
- `get_queries_from_questions`：查询规划  
- `fabricate_evidence_for_qa`：证据生成  

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

3.  **w/o Query Planning (IF2F)**:
    *   参数: `--no-concat-query`
    *   影响: 对于 IF2F，禁用查询感知证据生成，退回到原始的 `fabricate_evidence_for_qa` 方法。

---

## 5. IF2F 改进：轻量问题剪枝（低成本结构内优化）

IF2F（Improved Fact2Fiction）仍保持 Fact2Fiction 的 Planner/Executor 结构，但增加**轻量问题剪枝**，在不改变整体架构的情况下减少 API 调用并提高证据相关性。

### 5.1 改进动机

Fact2Fiction 的成本主要来自以下两步的 LLM 调用数量：

- **重要性评估**（`infer_qa_weight`）：每个问题一次调用
- **查询生成**（`get_queries_from_questions`）：每个问题一次调用

如果问题质量不高或与 justification 相关性不足，这些调用既消耗预算，也降低证据的命中率。

### 5.2 改进方案

**核心思想**：在 Answer Planning 后进行**本地相关性筛选**，保留最相关的子问题，减少后续 LLM 调用数量。

| 步骤 | 变化 |
|------|------|
| Answer Planning | 保持不变 |
| Question Pruning | 新增（无 API 成本） |
| Budget / Query / Evidence | 只对剪枝后的问题执行 |

### 5.3 技术实现（混合评分）

**核心函数**：`create_if2f_attack`

```python
def extract_entities(text):
    # 自定义规则扩展：专有名词短语、缩写、数字、引号内片段
    entities = set()
    entities.update(re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text))
    entities.update(re.findall(r"\b[A-Z]{2,}\b", text))
    entities.update(re.findall(r"\b\d{2,}\b", text))
    entities.update(re.findall(r"\"([^\"]{3,})\"", text))
    return set(e.strip() for e in entities if e.strip())

# 混合评分：BM25 相似度 + 实体重叠
context_text = claim_text + " " + (justification or "")
questions = [qa["question"] for qa in bad_qa_pairs]

bm25_scores = bm25(context_text, questions)  # 基于 query/context 的 BM25
context_entities = extract_entities(context_text)
entity_scores = [
    len(extract_entities(q) & context_entities) / max(1, len(extract_entities(q)))
    for q in questions
]

hybrid_scores = [
    0.7 * bm25_scores[i] + 0.3 * entity_scores[i]
    for i in range(len(questions))
]
scored = list(zip(bad_qa_pairs, hybrid_scores))
scored.sort(key=lambda x: x[1], reverse=True)
bad_qa_pairs = [qa for qa, _ in scored[:max_questions]]
```

### 5.4 策略优势

1. **API 成本降低**
   - 减少 `infer_qa_weight` 和 `get_queries_from_questions` 调用次数
   - 总调用量随剪枝比例下降

2. **证据相关性提高**
   - 问题更贴近 claim 和 justification
   - 伪造证据更集中在关键事实点

3. **结构不变**
   - 依然保留 Planner/Executor 全流程
   - 仅在中间增加本地筛选逻辑

### 5.5 预算控制

剪枝只减少问题数量，**不增加额外调用**。  
- 原方法：最多 10 个问题 → 10 次重要性评估 + 10 次查询生成  
- 新方法：问题数量压缩至 4~8 个（随 `num_fake` 调整）

### 5.6 参数控制

```bash
# 使用轻量剪枝（默认）
python attack/main.py --attack-type if2f ...

# 使用上一版本的剪枝方式（token overlap）
python attack/main.py --attack-type if2f --if2f-prune-method token

# 关闭轻量剪枝（退回原始问题数量）
python attack/main.py --attack-type if2f --no-if2f-prune ...

# 仍可关闭查询拼接
python attack/main.py --attack-type if2f --no-concat-query ...
```

---

## 6. Naive 攻击方法

Naive 攻击是一种最简单的基线攻击方法，它不依赖大模型生成复杂的文本，而是通过简单的规则直接生成否定的声明。

*   **函数**: create_naive_attack
*   **机制**:
    1.  获取原始声明 (Claim Text) 和原始结论 (Original Verdict)。
    2.  构造伪造证据：
        *   如果原结论是 **Refuted** (已反驳)，则直接重复原声明。逻辑是：既然原声明是错的，重复它就是在传播错误信息。
        *   如果原结论是 **Supported** (已支持)，则构造一个句子声明该 Claim 是错误的。
            *   模板: "The claim: '[Claim Text]' is [Negative Word]"
            *   [Negative Word] 随机选自: "inaccurate", "false", "fake", "hoax", "misleading" 等。
    3.  重复上述过程直到达到指定的伪造数量 (num_fake)。

这种方法生成的证据语义非常直白，缺乏论证细节，通常很难欺骗复杂的 Fact-Checking 系统，主要用作其他高级攻击方法的**下限基准 (Lower Bound / Baseline)**。
