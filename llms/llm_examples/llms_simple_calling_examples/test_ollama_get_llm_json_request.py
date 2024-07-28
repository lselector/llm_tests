import ollama
import json

def patched_request_stream(self, *args, stream=False, **kwargs):
    if args[0] == "POST" and args[1] == "/api/chat":
        json_data = kwargs.get('json', {})
        print(f"Raw JSON request:\n{json.dumps(json_data, indent=4)}")  # Pretty-print the JSON
    
    # Call the original _request_stream method
    return original_request_stream(self, *args, stream=stream, **kwargs) 

# Store the original _request_stream method
original_request_stream = ollama.Client._request_stream

# Monkey patch the _request_stream method of ollama.Client
ollama.Client._request_stream = patched_request_stream

# Now use the ollama client as usual
client = ollama.Client()

model = "llama3.1"

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
print(f"Answer:\n{response['message']['content']}")
