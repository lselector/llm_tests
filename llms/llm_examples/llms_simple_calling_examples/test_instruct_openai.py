
"""
# simple test on OpenAI API
"""

from openai import OpenAI
client = OpenAI()

response = client.completions.create(
  model="gpt-3.5-turbo-instruct",
  prompt="What is the capital city of France? Please answer in one word"
)

print(response.choices[0].text)
