
"""
# simple example ollama
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig

from langchain_openai import ChatOpenAI

# --------------------------------------------------------------
def get_answer(question):
    """ simple test """
    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You're a helpful assitant."),
         ("user", "{question}")]
    )

    runnable = prompt | llm

    response = runnable.invoke(question, config=RunnableConfig(callbacks=[]))
    print(response.content)

get_answer("why sky is blue?")