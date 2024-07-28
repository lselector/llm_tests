
"""
# test example running ollama with one of several models
"""

import ollama
client = ollama.Client()

model = "qwen2:7b-instruct"
# model = "mistral:instruct"
# model = "llama3:instruct"

prompt = "What is the capital city of France? Please answer in one word"
# prompt = "What I have just asked you?"

response = client.generate(model=model, prompt=prompt)
print(response['response'])
