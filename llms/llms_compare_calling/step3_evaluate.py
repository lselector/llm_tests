
"""
# step3_evaluate.py
# Usage:  python step3_evaluate.py
# This script evaluates the quality of LLM responses
# when calling them via Ollama using different methods:
#    1. chat system message
#    2. chat user message
#    3. "generate", no system message (instruct style)
#    4. "generate" with sysem message (instruct style + system)
# Evaluation scores are from 1 to 10 (higher - better)
# This script uses pure ollama and openai (no langchain)
# It reads responses from directory work/step2_answers/
# and write evaluations into work/step3_evaluation/
"""
import json, re
from openai import OpenAI
from string import Template
from mybag import *
from myutils import *

#---------------------------------------------------------------
def create_eval_prompt(bag): 
    """ creates openai prompt to evaluate accuracy of the compared LLMs """ 
    
    tt = Template("""
        You are very serious news llms accuracy tester.
        You are such a serious that you print only answer, nothing else.
                  
        Your main task is to compare llm1 answer, llm2 answer, llm3 answer and llm4 answer
        measure their accuracy to asked Question from 1 to 10 score. 
        Then explain why you put this score to this llm

        You have a list of firm rules that you always follow:
            1. You analyze the Question
            2. You analyze the llm1 answer, llm2 answer, llm3 answer and llm4 answer
            3. You compare these answers to Question and measure their accuracy to asked Question from 1 to 10 score
            4. Print the answer. Use pattern:
                  LLM1: 1
                  LLM2: 1
                  LLM3: 1
                  LLM4: 1
                  Explanation: explanation

        Question:
        $question
                  
        llm1 answer:
        $llm_1
                  
        llm2 answer:
        $llm_2
                  
        llm3 answer:
        $llm_3
                  
        llm4 answer:
        $llm_4
        """)

    tt = tt.substitute(question=bag.asked_question, 
                       llm_1=bag.answer_chat_system, 
                       llm_2=bag.answer_chat_user,
                       llm_3=bag.answer_instruct_prompt,
                       llm_4=bag.answer_instruct_prompt_system
                       )

    return tt

#---------------------------------------------------------------
def read_json_llms_answers(bag):
    """ Load all llms answers """
    bag.script_dir = os.path.dirname(os.path.realpath(__file__))
    folder_path = bag.script_dir + "/work/step2_answers"
    bag.answers_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding="utf-8") as fh:
                    data = json.load(fh)
                    bag.answers_data.append(data)
            except FileNotFoundError:
                print(f"Error: File '{filepath}' not found.")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in file '{filepath}'.")

#---------------------------------------------------------------
def evaluate_answer(bag, message):
    """ initialize system message for OpenAI model """
    completion = bag.llm_openai_client.chat.completions.create(
      model=bag.model_openai,
      messages=[{"role": "system", "content": f"{message}"}]
    )
    bag.llm_openai = completion
    bag.evaluation = bag.llm_openai.choices[0].message.content

#---------------------------------------------------------------
def get_explanation(answer, llm_index):
    """ returns score explanation text by word 'Explanation:' """
    explanation = answer.split(llm_index, 1)[1].strip()
    return explanation

#---------------------------------------------------------------
def write_results_to_json_file(bag,idx):
    """ Write test results to JSON file """
    bag.script_dir = os.path.dirname(os.path.realpath(__file__))
    bag.dir_out = bag.script_dir + "/work/step3_evaluation"
    os.makedirs(bag.dir_out, exist_ok=True)
    clean_str1 = title_to_alphanum(bag.model_answers['llm_answers'])
    clean_str2 = title_to_alphanum(bag.model_openai)
    fname = (f"{bag.dir_out}/{idx+1}_test_model_{clean_str1}_tester_{clean_str2}.json")
    with open(fname, "w", encoding="utf-8") as fh:
        print(f"writing {fname}")
        json.dump(bag.test_results, fh, indent=4)

#---------------------------------------------------------------
def llm_accuracy_test_p1(bag):
    """ init result lists, called from llm_accuracy_test """
    bag.test_results = { "title": "Comparing: system message vs user (non system) message", 
                         "model_in_test": bag.model_answers['llm_answers'],
                         "model_tester" : bag.model_openai,
                         "llm_question_generator" : bag.model_answers['llm_question_generator'],
                         "average_system_message_score" : "",
                         "average_user_message_score" : "",
                         "average_instruct_prompt_score" : "",
                         "average_instruct_prompt_system_score" : "",
                         "tests": [] }
    
    bag.llm_1_result_list = []
    bag.llm_2_result_list = []
    bag.llm_3_result_list = []
    bag.llm_4_result_list = []

    print(" ====================== TEST START ===================================")
    print(" === System message vs user (non system) message =====================")
    print(f" === Tested model: {bag.model_answers['llm_answers']} ")
    print(f" === Tester: {bag.model_openai} \n")

#---------------------------------------------------------------
def print_q_and_a(bag):
    """ prints questions and answers, called from llm_accuracy_test() """
    print("\nAsked question:")
    print(f"\n{bag.asked_question}")
    print("\nAnswer from system message llm:")
    print(f"\n{bag.answer_chat_system}")
    print("\nAnswer from user (no system) message llm:")
    print(f"\n{bag.answer_chat_user}")
    print("\nAnswer from instruct prompt llm:")
    print(f"\n{bag.answer_instruct_prompt}")
    print("\nAnswer from instruct system + prompt llm:")
    print(f"\n{bag.answer_instruct_prompt_system}")
    print(f"\nEvaluated result: (Scores: 1 - 10) \n{bag.evaluation}")

# --------------------------------------------------------------
def save_test_info_to_bag(bag):
    """ append test info to bag.test_results["tests"], called from llm_accuracy_test """        

    pattern = r"LLM\d:\s*(\d{1,2})"  # Match "LLMn:" followed by 1 or 2 digits
    scores = re.findall(pattern, bag.evaluation)  # Find all matches
    # Convert scores from strings to integers
    scores = [int(score) for score in scores]
    num1 = scores[0]
    num2 = scores[1]
    num3 = scores[2]
    num4 = scores[3]

    explanation = get_explanation(bag.evaluation, "Explanation:")

    bag.llm_1_result_list.append(num1)
    bag.llm_2_result_list.append(num2)
    bag.llm_3_result_list.append(num3)
    bag.llm_4_result_list.append(num4)

    test_info = {
                "test_id"                                  : bag.test_id,
                "asked_question"                           : bag.asked_question, 
                "answer_system_message_llm"                : bag.answer_chat_system, 
                "answer_user_message_llm"                  : bag.answer_chat_user, 
                "answer_instruct_prompt_llm"               : bag.answer_instruct_prompt,
                "answer_instruct_prompt_system_llm"        : bag.answer_instruct_prompt_system, 
                "answer_system_message_llm_score"          : num1,
                "answer_user_message_llm_score"            : num2,
                "answer_instruct_prompt_llm_score"         : num3,
                "answer_instruct_prompt_system_llm_score"  : num4,
                "explanation_of_scores"                    : explanation
            }
    
    bag.test_results["tests"].append(test_info)

#---------------------------------------------------------------
def llm_accuracy_test(bag):
    """ Compares and evaluates llm answers """
    llm_accuracy_test_p1(bag)

    for i in range(len(bag.model_answers['answers'])):
        bag.test_id = i
        print(f" === test started: {i} === model {bag.model_answers['llm_answers']} ===========================")

        bag.asked_question = bag.model_answers['answers'][i]['asked_question']
        bag.answer_chat_system = bag.model_answers['answers'][i]['answer_system_message_llm']
        bag.answer_chat_user = bag.model_answers['answers'][i]['answer_user_message_llm']
        bag.answer_instruct_prompt = bag.model_answers['answers'][i]['answer_instruct_prompt_llm']
        bag.answer_instruct_prompt_system = bag.model_answers['answers'][i]['answer_instruct_prompt_and_system_llm']

        print("Evaluate answers\n")
        prompt_for_llm_evaluator = create_eval_prompt(bag)
        evaluate_answer(bag, prompt_for_llm_evaluator) # sets bag.evaluation
        print_q_and_a(bag)
        save_test_info_to_bag(bag)
    # -------------------------------------------------
    # after evaluating all responses, we can calculate averages
    average1 = sum(bag.llm_1_result_list) / len(bag.llm_1_result_list) 
    bag.test_results["average_system_message_score"] = round(average1,2)
    average2 = sum(bag.llm_2_result_list) / len(bag.llm_2_result_list)
    bag.test_results["average_user_message_score"]   = round(average2,2)
    average3 = sum(bag.llm_3_result_list) / len(bag.llm_3_result_list)
    bag.test_results["average_instruct_prompt_score"] = round(average3,2)
    average4 = sum(bag.llm_4_result_list) / len(bag.llm_4_result_list)
    bag.test_results["average_instruct_prompt_system_score"] = round(average4,2)

# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------
bag = MyBunch()
bag.llm_openai_client = OpenAI()
bag.model_openai="gpt-4o"   # or "gpt-4-turbo"
read_json_llms_answers(bag)
for idx in range(5):
    for answers in bag.answers_data:
        bag.model_answers = answers
        llm_accuracy_test(bag)
        write_results_to_json_file(bag,idx)
print("\n\nDONE")
