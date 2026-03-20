import re

with open(r'H:\Fact2Fiction\src\attack\attack_utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the query_gpt_batch function body
old_body = """    results = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=5)) as session:
        tasks = []
        for _ in tqdm(range(max_limits), desc="Creating tasks"):
            tasks.append(query_gpt_single(session, input, model_name, max_tokens, temperature))
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Creating Fake Evidence with API"):
            result = await task
            results.append(result)
        return results"""

new_body = """    tasks = []
    for _ in tqdm(range(max_limits), desc="Creating tasks"):
        tasks.append(query_gpt_single(None, input, model_name, max_tokens, temperature))
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Creating Fake Evidence with API"):
        result = await task
        results.append(result)
    return results"""

if old_body in content:
    content = content.replace(old_body, new_body)
    print("Replaced query_gpt_batch body")
else:
    print("ERROR: old_body not found!")
    # Debug: find the function
    idx = content.find('async def query_gpt_batch')
    if idx >= 0:
        print("Function found at index", idx)
        print(repr(content[idx:idx+500]))

with open(r'H:\Fact2Fiction\src\attack\attack_utils.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Also remove the 'import aiohttp' if it's still there
with open(r'H:\Fact2Fiction\src\attack\attack_utils.py', 'r', encoding='utf-8') as f:
    content = f.read()
if 'import aiohttp' in content:
    content = content.replace('import aiohttp\n', '')
    with open(r'H:\Fact2Fiction\src\attack\attack_utils.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Removed aiohttp import")
else:
    print("aiohttp import already removed")
