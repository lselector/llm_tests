
""" 
# This script creates questions for llms calls evaluation 
"""

import json, argparse
from openai import OpenAI
from mybag import *

#---------------------------------------------------------------
def process_cmd_args(bag):
    """ define bag.parser arguments for CMD arguments """
    # -------------------------------------------
    description_str = """
        This script creates questions for llms calls evaluation
    """
    bag.parser = argparse.ArgumentParser(description=description_str)
    # ---- specify LLM - default gpt-4-turbo ----
    bag.parser.add_argument("-aa", "--aa", action='store_true',
        default=False, dest='arg_a',
        help="optional, use OpenAI service with gpt-4-turbo model. Will be used by default if no args" )
    bag.parser.add_argument("-oo", "--oo", action='store_true',
        default=False, dest='arg_o',
        help=f"optional, use service with gpt-4o model" )
    # -------------------------------------------
    bag.parsed, unknown = bag.parser.parse_known_args()
    if len(unknown):
        print("unrecognized argument(s) " + str(unknown))
        sys.exit(1)
    # -------------------------------------------
    bag.arg_service = "openai"
    bag.arg_a = False
    bag.arg_o = False
    bag.model_openai = "gpt-4-turbo"
    if bag.parsed.arg_o:
        bag.model_openai = "gpt-4o"
    bag.model_type = bag.arg_service + "--" + bag.model_openai

#---------------------------------------------------------------
def create_prompt_make_task_openai(): 
    """ creates a prompt to create a question to evaluate accuracy of the LLM model """ 
    ss = """
        You are very serious news llms accuracy tester.
        You are such a serious that you print only answer, nothing else.
                  
        Your main task is to create a question for llm accuracy test and print this question as answer.

        You have a list of firm rules that you always follow:
            1. You create the question for llm testing:
                  This should be complex question to do a stress test for llm
            2. Print this question as answer
            3. You always use pattern to generate question:
                  Question:
                  ...

                  Insturctions how to deal with this question:
                  1. ...
                  2. ...
                  3. ...
                  ...
        """

    return ss  

#---------------------------------------------------------------
def create_one_question(bag, message):
    """ calls openai to create one question """
    completion = bag.llm_openai_client.chat.completions.create(
      model=bag.model_openai,
      messages=[ {"role": "system", "content": f"{message}"} ]
    )
    bag.response = completion

#---------------------------------------------------------------
def create_questions(bag, model):
    """ 
    # Creates questions for LLMS, 
    # populates bag.test_results["questions"] 
    """
    # -------------------------------------------
    bag.test_results = { 
        "title"                  : "Questions for llms", 
        "llm_question_generator" : f"{bag.model_openai}",
        "questions"              : [] 
    }
    print(f" ")
    print(f" ==================== Questions Generation Start =====================")
    print(f" === Questions for LLMS accuracy competition     =====================")
    print(f" === Questions generator: {bag.model_openai} \n")
    # -------------------------------------------
    prompt_q_maker = create_prompt_make_task_openai()
    
    for i in range(bag.n_iterations):
        print(f" === Question generating: {i} ===============")

        create_one_question(bag, prompt_q_maker)
        bag.question_for_llm = bag.response.choices[0].message.content
        print(f"Generated question: \n{bag.question_for_llm}\n")

        parts = bag.question_for_llm.split("\n\n")  # Split by two newlines

        # Assigning to variables
        question = parts[0].strip()  # Remove any extra whitespace
        instructions = parts[1].strip()

        test_info = {
            "question_id"  : i,
            "question"     : question,
            "instructions" : instructions
        }
        
        bag.test_results["questions"].append(test_info)

#---------------------------------------------------------------
def write_results_to_json_file(bag):
    """ Write bag.test_results to JSON file """
    bag.script_dir = os.path.dirname(os.path.realpath(__file__))
    bag.dir_out = bag.script_dir + "/work/step1_questions"
    os.makedirs(bag.dir_out, exist_ok=False)
    fname = (f"{bag.dir_out}/questions.json")
    with open(fname, "w", encoding="utf-8") as fh:
        print(f"writing {fname}")
        json.dump(bag.test_results, fh, indent=4)

# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------
bag = MyBunch()
bag.n_iterations = 30
process_cmd_args(bag)
bag.llm_openai_client = OpenAI()
create_questions(bag, bag.model_openai)
write_results_to_json_file(bag)
