from openai import OpenAI
from pydantic import BaseModel
import os
os.environ.pop("OPENAI_BASE_URL", None)
os.environ.pop("OPENAI_API_BASE", None)

class Step(BaseModel):
    explanation: str
    output: str

client = OpenAI(base_url="http://localhost:8010/v1", api_key="EMPTY")

sp = ''
i = '''Write a function to that returns true if the input string contains sequences of lowercase letters joined with an underscore and false otherwise.\n\nImplement the following Python function signature exactly:\ndef text_lowercase_underscore(text):\n\nPlease think step by step, and return the python code start with `def` and be executable Python code.'''



resp = client.chat.completions.parse(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": i}],
    temperature=0.0,
    response_format = Step
)

print(resp)
# print(resp.choices[0].message.content)
