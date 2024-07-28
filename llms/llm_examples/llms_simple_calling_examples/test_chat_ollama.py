import ollama

# Use local Ollama server
client = ollama.Client()

model = "llama3"

prompt = "What is the capital city of France?"
system_message = """ 
You are a helpful and concise assistant. 
You always say 'nya' at the end of an answer.
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": prompt},
]

response = client.chat(model=model, messages=messages)
print(response['message']['content'])
