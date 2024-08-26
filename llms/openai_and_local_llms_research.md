## OpenAI and Local LLMs research notes

To became more accurate output we need to use messages with role "System".
--------

#### There are three ways to use system message for Ollama models:

- Use Sytem messages to Ollama through LangChain

    Example:

    ```python

    from langchain.schema.runnable.config import RunnableConfig

    from langchain_community.llms import Ollama
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import StrOutputParser
    from langchain.schema.runnable.config import RunnableConfig


    def get_answer(question):

        model = Ollama(model="llama3")
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You're a helpful assitant. {question}")]
        )
        runnable = prompt | model | StrOutputParser()

        response = runnable.invoke(question, config=RunnableConfig(callbacks=[]))
        print(response)
        return response

    get_answer("What is the capital of Ukraine?")

    ```

    Example 2:

    ```python
    
    from langchain_community.chat_models import ChatOllama
    from langchain.schema.runnable.config import RunnableConfig
    from langchain.schema import (
        SystemMessage,
        HumanMessage,
    )
    
    chat = ChatOllama(model="llama3")
    
    messages = [
        SystemMessage(content="You are a helpful and informative AI assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    response = chat.invoke(messages, config=RunnableConfig(callbacks=[]))
    print(response.content)
    ```

    Example of using system message with OpenAI. It's very similar to 1st example with Ollama
    
    ```python

    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable.config import RunnableConfig

    from langchain_openai import ChatOpenAI

    def openai_init(question):

        llm = ChatOpenAI(model="gpt-4o")
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You're a helpful assitant. {question}",)]
        )

        runnable = prompt | llm

        response = runnable.invoke(question, config=RunnableConfig(callbacks=[]))
        print(response.content)

    openai_init("why sky is blue?")
    ```


- If there is no way to use langchain first message you send when creating a chat session should automatically treated as the system message

- Also we can emulate system message when we put our System message on top of the prompt.

    Example:

    ```python
    prompt = """

    Task: 
    1.Summaraize next Text in 3 - 5 sentences.
    2.Use Taras Shevchenko's writing style.
    3.Do not miss any details.

    Text:

    """
    ```

### OpenAI prompt experiments

#### Experiment with multitasking

There were done 2 experiments with using OpenAI API (GPT4o)

llm_combiner has been takes as experimental task.

Inside the combiner we have 2 main tasks:

- deduplication
- ranking

I did deduplication using 2 different methods:

- prompt to return only list with duplicates
    This method works a little faster
    It finds the duplicates better

- prompt to return deduplicated list
    We have only one call that do everything
    Do not detects all the duplicates, works worth than 1st method
    To do answer stable and quick need to print really good prompt with a lot of steps in details how to do the tasks.

#### Summary:
One task at a time works more reliable. But if we have big enogh model and very good prompt
Multitask (prompt to return deduplicated list) will work as single task.

The main issue for now - not the 1st nor the second method do not recognize all the duplicates.
Need to do experiments with prompt or search for another method (like vectorDB)
