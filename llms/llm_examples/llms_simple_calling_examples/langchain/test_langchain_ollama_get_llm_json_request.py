from typing import List, Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.schema.runnable.config import RunnableConfig

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

import json

class CaptureLLMInputCallback(BaseCallbackHandler):
    def __init__(self):
        self.llm_inputs = []

    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs: Any) -> None:
        self.llm_inputs.append(serialized)  

def get_answer(question):
    model = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You're a helpful assistant"),
         ("user", "{question}")]
    )

    callback_handler = CaptureLLMInputCallback()

    chain = prompt | model | StrOutputParser()
    response = chain.invoke(question, config=RunnableConfig(callbacks=[callback_handler]))

    # Get both the formatted prompt and the final LLM input JSON
    formatted_prompt = prompt.format(question=question)
    final_llm_input_json = callback_handler.llm_inputs[0]

    # Create the LLM interaction dictionary
    llm_interaction = {
        "formatted_prompt": formatted_prompt,
        "llm_input_json": final_llm_input_json,
        "output": response
    }

    # Optional: Save to JSON file
    print(f"RAW JSON request: {json.dumps(llm_interaction, indent=4)}")

    print(response)
    return response

# Example usage
get_answer("What is the capital of Ukraine?")
