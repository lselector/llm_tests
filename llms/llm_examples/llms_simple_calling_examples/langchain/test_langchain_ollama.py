
"""
# simple example ollama
"""

from langchain.schema.runnable.config import RunnableConfig

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig

# --------------------------------------------------------------
def get_answer(question):
    """ simple test """
    model = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You're a helpful assitant"),
         ("user", "{question}")]
    )
    
    runnable = prompt | model | StrOutputParser()
    response = runnable.invoke(question, config=RunnableConfig(callbacks=[]))
    print(response)
    return response

# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------
get_answer("What is the capital of Ukraine?")
