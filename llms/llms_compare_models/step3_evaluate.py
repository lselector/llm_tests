
"""
# step3_evaluate.py - script to evaluate quality of LLM response
# when calling it via Ollama using different methods:
#    1. chat system message
#    2. chat user message
#    3. "generate", no system message (instruct style)
#    4. "generate" with sysem message (instruct style + system)
# Evaluation scores are from 1 to 10 (higher - better)
# This script uses pure ollama and openai (no langchain)
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
                  
Your main task is to compare $llm_names
measure their accuracy to asked Question from 1 to 10 score. 
Then explain why you put this score to this llm

You have a list of firm rules that you always follow:
    1. You analyze the Question
    2. You analyze the $llm_names
    3. You compare these answers to Question and measure their accuracy to asked Question from 1 to 10 score
    4. Print the answer. Use pattern:
$score_pattern
        Explanation: explanation

        $question
                  
        """)
    
    llm_names = ", ".join(bag.names_answers.keys())
    score_pattern = "\n".join([f"\t{llm}: 1" for llm in bag.names_answers])

    # Create the initial prompt without the answer_pattern
    prompt = tt.substitute(
        llm_names=llm_names,
        score_pattern=score_pattern,
        question=bag.asked_question,
    )

    # Now, append the answer_pattern separately for each LLM
    for llm, answer in bag.names_answers.items():
        prompt += f"\n\n{llm} answer: ============================================== \n{answer}"

    return prompt

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
def write_results_to_json_file(bag, iterate):
    """ Write test results to JSON file """
    bag.script_dir = os.path.dirname(os.path.realpath(__file__))
    bag.dir_out = bag.script_dir + "/work/step3_evaluation"
    os.makedirs(bag.dir_out, exist_ok=True)
    models_names = ""
    for model_name in bag.llms_real_names:
        models_names += model_name + "_"
    str_clear1 = title_to_alphanum(models_names)
    str_clear2 = title_to_alphanum(bag.model_openai)
    fname = (f"{bag.dir_out}/{iterate}_tested_models_{str_clear1}_tester_{str_clear2}.json")
    with open(fname, "w", encoding="utf-8") as fh:
        print(f"writing {fname}")
        json.dump(bag.test_results, fh, indent=4)

#---------------------------------------------------------------
def llm_accuracy_test_p1(bag):
    """ init result lists, called from llm_accuracy_test """
    bag.test_results = { "title": "Comparing llms by answers accuracy", 
                         "models_in_test": bag.llms_real_names,
                         "model_tester" : bag.model_openai,
                         "llm_question_generator" : bag.answers_data[0]['llm_question_generator']
                         }
    
    get_lists_of_scores(bag)
    for idx in range(len(bag.lists_of_scores)):
           bag.test_results[f"average_{bag.llms_real_names[idx]}_score"] = ""

    bag.test_results["tests"] = [] 

    print(" ====================== TEST START ===================================")
    print(" === Comparing llms by answers accuracy          =====================")
    print(f" === Tested models: {bag.llms_real_names} ")
    print(f" === Tester: {bag.model_openai} \n")

#---------------------------------------------------------------
def print_q_and_a(bag):
    """ prints questions and answers, called from llm_accuracy_test() """
    print("\nAsked question:")
    print(f"\n{bag.asked_question}")

    for idx in range(len(bag.llms_real_names)):
        print(f"\n Answer from {bag.llms_real_names[idx]}")
        buff = f"LLM{idx+1}"
        print(f"\n {bag.names_answers[buff]}")

    print(f"\nEvaluated result: (Scores: 1 - 10) \n{bag.evaluation}")

#---------------------------------------------------------------
def get_lists_of_scores(bag):
    """ Creates empty list of lists where will be putted scores """
    bag.lists_of_scores = []

    for _ in range(len(bag.llms_real_names)):
        list_of_scores = []
        bag.lists_of_scores.append(list_of_scores)

# --------------------------------------------------------------
def save_test_info_to_bag(bag):
    """ append test info to bag.test_results["tests"], called from llm_accuracy_test """        

    pattern = r"LLM\d:\s*(\d{1,2})"  # Match "LLMn:" followed by 1 or 2 digits
    scores = re.findall(pattern, bag.evaluation)  # Find all matches
    # Convert scores from strings to integers
    scores = [int(score) for score in scores]

    for idx in range(len(bag.lists_of_scores)):
            bag.lists_of_scores[idx].append(scores[idx])
    explanation = get_explanation(bag.evaluation, "Explanation:")

    test_info = {
                "test_id"                                  : bag.test_id,
                "asked_question"                           : bag.asked_question               
            }
    
    for idx in range(len(scores)):
        test_info[f"answer_{bag.llms_real_names[idx]}"] = scores[idx]

    for idx in range(len(bag.llms_real_names)):
        test_info[f"answer_from_{bag.llms_real_names[idx]}"] = bag.names_answers[f"LLM{idx+1}"]
    
    test_info["explanation_of_scores"] = explanation
    
    bag.test_results["tests"].append(test_info)

#---------------------------------------------------------------
def llm_accuracy_test(bag):
    """ Compares and evaluates llm answers """
    llm_accuracy_test_p1(bag)

    bag.names_answers = {}

    for idx in range(len(bag.answers_data[0]['answers'])):
        bag.test_id = idx
        bag.asked_question = bag.answers_data[0]['answers'][idx]['asked_question']
        print(f" === test started: {idx} === models {bag.llms_real_names} ===\n")
        for jdx in range(len(bag.answers_data)):

            bag.names_answers[bag.llms_names[jdx]] = bag.answers_data[jdx]['answers'][idx]['answer_instruct_prompt_llm']

        print("Evaluate answers\n")
        prompt_for_llm_evaluator = create_eval_prompt(bag)
        evaluate_answer(bag, prompt_for_llm_evaluator) # sets bag.evaluation
        print_q_and_a(bag)
        save_test_info_to_bag(bag)
    # -------------------------------------------------
    # after evaluating all responses, we can calculate averages

    for idx in range(len(bag.lists_of_scores)):
           average = sum(bag.lists_of_scores[idx]) / len(bag.lists_of_scores[idx])
           bag.test_results[f"average_{bag.llms_real_names[idx]}_score"] = round(average,2)

# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------
bag = MyBunch()
bag.llm_openai_client = OpenAI()
bag.model_openai="gpt-4o"   # or "gpt-4-turbo"
read_json_llms_answers(bag)
for iterate in range(5):
    bag.llms_names = []
    bag.llms_real_names = []
    for idx in range(len(bag.answers_data)):
        bag.llms_names.append(f"LLM{idx+1}")
        bag.llms_real_names.append(bag.answers_data[idx]['llm_answers'])

    llm_accuracy_test(bag)
    write_results_to_json_file(bag, iterate)
print("\n\nDONE")
