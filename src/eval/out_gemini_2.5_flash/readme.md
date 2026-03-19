Claim-Before    声明 与 攻击前 justification 的余弦相似度（均值 +/- 标准差）。这是基线，反映系统正常输出的 justification 与声明有多相关
Claim-After	    声明 与 攻击后 justification 的余弦相似度。反映攻击后 justification 是否还在紧密围绕声明论述
Before-After    攻击前 justification 与 攻击后justification 的余弦相似度。


维度 A: 恶意语料内容特征
编码	类别	说明
A1	Fact Fabrication	完全捏造不存在的事实、事件或声明
A2	Fact Distortion	基于真实事件，篡改关键细节以反转含义
A3	False Attribution	将不存在的行为/言论归因于真实的权威机构
A4	Fabricated Data	捏造具体的数字、日期或统计数据
A5	Context Manipulation	将真实信息放在错误上下文中以改变含义
A6	Direct Instruction Injection	显式指令告诉模型输出特定结论

维度 B: 攻击成功机制
编码	类别	说明
B1	Evidence Monopolization	虚假证据完全占据检索结果，阻断真实证据
B2	Evidence Contamination	虚假证据与真实证据混合，混淆系统
B3	Reasoning Chain Reconstruction	针对每个子问题构建完整的替代叙事
B4	Key Evidence Displacement	替换支持正确结论的关键证据
B5	Authority Deception	模仿权威来源的格式和语气

维度 C: 利用的系统漏洞
编码	类别	说明
C1	Retrieval Trust Blindness	系统无条件信任所有检索到的证据
C2	Quantity Over Quality	大量虚假证据在数量上压制真实证据
C3	Missing Source Verification	系统未验证证据来源的可信度
C4	Insufficient Consistency Check	系统未检测逻辑矛盾
C5	Inherent Claim Ambiguity	声明本身存在