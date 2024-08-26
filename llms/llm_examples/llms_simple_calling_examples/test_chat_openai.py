
"""
# simple test on OpenAI API
"""

from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "What is the Capital city of France? Answer in one word."},
    {"role": "user", "content": "Answer in one word."}
  ]
)

print(completion.choices[0].message.content)