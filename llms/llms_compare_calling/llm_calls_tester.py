
"""
# There is one model that we can use for chat and instructions - Instruct
# There are 3 methods to do call to a model:
    - Instruct style where we have common a message and do not have system message
      (when we use this style we have system messages but we can put there only
      instructions. No way to put task, or llm will freeze)
    - Chat style with system messages
    - Chat style with user messages

# This script compares all call styles
# Evaluates them from 1 to 10 score
# Prints explanation of evaluation

This script use pure ollama and openai (no langchain)
"""
import ollama, json, re
from openai import OpenAI
from string import Template

from mybag import *

#---------------------------------------------------------------
def create_prompt_evaluate_llms_responces_openai(bag): 
    """ creates a prompt to evaluate accuracy of the compared LLMs """ 
    
    tt = Template("""
        You are very serious news llms accuracy tester.
        You are such a serious that you print only answer, nothing else.
                  
        Your main task is to compare llm1 answer, llm2 answer and llm3 answer,
        measure their accuracy to asked Question from 1 to 10 score. 
        Then explain why you put this score to this llm

        You have a list of firm rules that you always follow:
            1. You analyze the Question
            2. You analyze the llm1 answer and llm2 answer
            3. You compare these answers to Question and measure their accuracy to asked Question from 1 to 10 score
            4. Print the answer. Use pattern:
                  LLM1: 1
                  LLM2: 1
                  LLM3: 1
                  Explanation: explanation

        Question:
        $question
                  
        llm1 answer:
        $llm_1
                  
        llm2 answer:
        $llm_2
                  
        llm2 answer:
        $llm_3
        """)
    

    tt = tt.substitute(question=bag.question_for_llm, 
                       llm_1=bag.asnwer_llm_system_message, 
                       llm_2=bag.asnwer_llm_user_message,
                       llm_3=bag.asnwer_llm_instruct
                       )

    return tt

#---------------------------------------------------------------
def create_prompt_make_task_openai(): 
    """ creates a prompt to create a question to evaluate accuracy of the LLM model """ 
    
    tt = Template("""
        You are very serious news llms accuracy tester.
        You are such a serious that you print only answer, nothing else.
                  
        Your main task is to create a question for llm accuracy test and print this question as answer.

        You have a list of firm rules that you always follow:
            1. You create the question for llm testing:
                  This should be complex question to do a stress test for llm
            2. Print this question as answer
        """)

    return tt.template

#---------------------------------------------------------------
def init_ollama_and_openai_models(bag):
    """ initialize Ollama and OpenAI models """
    bag.llm_ollama = ollama.Client()
    bag.llm_openai_client = OpenAI()

#---------------------------------------------------------------
def init_chat_system_ollama_llm(bag, message):
    """ initialize system message for Ollama model """
    messages = [
        {"role": "system", "content": message + " \nLimit responce to 100 tokens."}
    ]

    bag._ollama_system_message = messages

#---------------------------------------------------------------
def init_chat_user_ollama_llm(bag, message):
    """ initialize user message for Ollama model """
    messages = [
        {"role": "user", "content": message + " \nLimit responce to 100 tokens."}
    ]

    bag._ollama_user_message = messages

#---------------------------------------------------------------
def init_instruct_llm(bag, message):
    """ initialize instruct message for Ollama model """
    bag._ollama_instruct_message = message

#---------------------------------------------------------------
def init_llm_tester_openai(bag, message):
    """ initialize system message for OpenAI model """
    completion = bag.llm_openai_client.chat.completions.create(
      model=bag.model_openai,
      messages=[
        {"role": "system", "content": f"{message}"}
      ]
    )

    bag.llm_openai = completion

#---------------------------------------------------------------
def get_explanation(answer, llm_index):
    """ returns score explanation text by word 'Explanation:' """
    explanation = answer.split(llm_index, 1)[1].strip()
    return explanation

def write_results_to_json_file():
    """ Write test results to JSON file """
    bag.script_dir = os.path.dirname(os.path.realpath(__file__))
    bag.dir_out = bag.script_dir + "/system_message_test"
    os.makedirs(bag.dir_out, exist_ok=True)
    fname = (f"{bag.dir_out}/test_model_{bag.model_ollama}_tester_{bag.model_openai}.json")
    with open(fname, "w", encoding="utf-8") as fh:
        print(f"writing {fname}")
        json.dump(bag.test_results, fh, indent=4)

# --------------------------------------------------------------
def get_llm_score_index(text, llm_index):
    """ gives index of llm score """
    index = text.find(llm_index)
    return index+5

# --------------------------------------------------------------
def get_llm_score(answer, llm_index):
    """ retrieves llm score by word LLM_1 ... LLM_N """
    index =  get_llm_score_index(answer, llm_index)
    isSecondValue = None
    try:
        isSecondValue = int(answer[index + 1])
    except TypeError:
        return int(answer[index])

    if isSecondValue:
        num = answer[index] + answer[index + 1]
        return int(num)

#---------------------------------------------------------------
def llm_accuracy_test(bag):
    """ Does the comparing and evaluating of llm answers """

    init_ollama_and_openai_models(bag)

    bag.test_results = { "title": "Comparing: system message vs user (non system) message", 
                         "model_in_test": f"{bag.model_ollama}",
                         "model_tester" : f"{bag.model_openai}",
                         "average_system_message_score" : "",
                         "average_user_message_score" : "",
                         "average_instruct_score" : "",
                         "tests": [] }
    
    llm_1_result_list = []
    llm_2_result_list = []
    llm_3_result_list = []

    print(" ====================== TEST START ===================================")
    print(" === System message vs user (non system) message =====================")
    print(f" === Tested model: {bag.model_ollama} ")
    print(f" === Tester: {bag.model_openai} \n")

    for i in range(bag.iteratives):
        print(f" ====================== Test started: {i} ==============================")
        print("Creating qustion")
        prompt_for_llm_question_maker = create_prompt_make_task_openai()

        init_llm_tester_openai(bag, prompt_for_llm_question_maker)
        bag.question_for_llm = bag.llm_openai.choices[0].message.content

        init_chat_system_ollama_llm(bag, bag.question_for_llm)
        print("Ask question through system message")
        bag.asnwer_llm_system_message = bag.llm_ollama.chat(model=bag.model_ollama, messages=bag._ollama_system_message)['message']['content']

        init_chat_user_ollama_llm(bag, bag.question_for_llm)
        print("Ask question through user (non system) message")
        bag.asnwer_llm_user_message = bag.llm_ollama.chat(model=bag.model_ollama, messages=bag._ollama_user_message)['message']['content']

        init_instruct_llm(bag, bag.question_for_llm)
        print("Ask question using instruct style")
        bag.asnwer_llm_instruct = bag.llm_ollama.generate(model=bag.model_ollama, prompt=bag._ollama_instruct_message)['response']
        
        print("Evaluate answers\n")
        prompt_for_llm_evaluator = create_prompt_evaluate_llms_responces_openai(bag)

        init_llm_tester_openai(bag, prompt_for_llm_evaluator)
        answer = bag.llm_openai.choices[0].message.content

        print("\nAsked question:")
        print(f"\n{bag.question_for_llm}")
        print("\nAnswer from system message llm:")
        print(f"\n{bag.asnwer_llm_system_message}")
        print("\nAnswer from user (no system) message llm:")
        print(f"\n{bag.asnwer_llm_user_message}")
        print("\nAnswer from instruct style llm:")
        print(f"\n{bag.asnwer_llm_instruct}")
        print(f"\nEvaluated result: (Scores: 1 - 10) \n{answer}")

        #num1 = get_llm_score(answer, "LLM1:")
        #num2 = get_llm_score(answer, "LLM2:")
        #num3 = get_llm_score(answer, "LLM3:")

        pattern = r"LLM\d:\s*(\d{1,2})"  # Match "LLMn:" followed by 1 or 2 digits

        scores = re.findall(pattern, answer)  # Find all matches

        # Convert scores from strings to integers
        scores = [int(score) for score in scores]
        num1 = scores[0]
        num2 = scores[1]
        num3 = scores[2]

        explanation = get_explanation(answer, "Explanation:")

        llm_1_result_list.append(num1)
        llm_2_result_list.append(num2)
        llm_3_result_list.append(num3)

        test_info = {
                    "test_id"                         : i,
                    "asked_question"                  : bag.question_for_llm, 
                    "answer_system_message_llm"       : bag.asnwer_llm_system_message, 
                    "answer_user_message_llm"         : bag.asnwer_llm_user_message, 
                    "answer_instruct_llm"             : bag.asnwer_llm_instruct, 
                    "answer_system_message_llm_score" : num1,
                    "answer_user_message_llm_score"   : num2,
                    "answer_instruct_llm_score"       : num3,
                    "explanation_of_scores"           : explanation
                }
        
        bag.test_results["tests"].append(test_info)

    average1 = sum(llm_1_result_list) / len(llm_1_result_list) 
    bag.test_results["average_system_message_score"] = average1
    average2 = sum(llm_2_result_list) / len(llm_2_result_list)
    bag.test_results["average_user_message_score"] = average2
    average3 = sum(llm_3_result_list) / len(llm_3_result_list)
    bag.test_results["average_instruct_score"] = average3

# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------
bag = MyBunch()

models_to_test = ["llama3:instruct",
                  "mistral:7b-instruct",
                  "qwen2:7b-instruct"]

#bag.model_openai="gpt-4o"
bag.model_openai="gpt-4-turbo"

bag.iteratives = 30

for model in models_to_test:

    bag.model_ollama = model

    llm_accuracy_test(bag)
    write_results_to_json_file()

print("\n\nDONE")
