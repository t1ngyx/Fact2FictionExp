from openai import OpenAI

client = OpenAI(
    api_key="sk-Rdant1aG5r25RRALMoYKCtQKIWc32qG6U8mztGa8nntVdKw1",
    base_url="https://www.chataiapi.com/v1",
)

completion = client.chat.completions.create(
    model="gemini-2.5-flash",  # 修改为 gemini-1.5-flash，如果是2.0版本可用 gemini-2.0-flash-exp
    messages=[{"role": "user", "content": "1+1等于多少？请给出详细的思考过程，然后给出最终答案。"}],
)

print("思考过程：")
# Gemini 模型通常没有 reasoning_content 字段，添加 getattr 避免报错
print(getattr(completion.choices[0].message, "reasoning_content", "无思考过程"))

print("最终答案：")
print(completion.choices[0].message.content)