import ollama

model_name = 'llama3.1'  # Replace with your desired model
messages = [ {'role': 'user', 'content': 'Why is the sky blue?'} ]

for chunk in ollama.chat(model=model_name, messages=messages, stream=True):
    print(chunk['message']['content'], end='', flush=True)

