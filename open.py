from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8010/v1", api_key="EMPTY")


import os
print(os.getenv("OPENAI_BASE_URL"))
print(os.getenv("OPENAI_API_BASE"))

resp = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "hello"}]
)

print(resp.choices[0].message["content"])
