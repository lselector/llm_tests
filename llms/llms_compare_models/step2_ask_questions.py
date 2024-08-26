
""" 
# This script asks questions to test models
"""

import ollama, json, argparse
from mybag import *
from myutils import *

#---------------------------------------------------------------
def read_questions_from_json_file(bag):
    """ 
    # reads a JSON news object and returns extracted relevant fields as a dict 
    # combining text elements into one string
    """
    file_path = "work/step1_questions/questions.json"
    with open(file_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    bag.questions_data = {
        'title'                   : data['title'],
        'llm_question_generator'  : data['llm_question_generator'],  # Combine into a single string
        'questions'               : data.get("questions", None)
    }

#---------------------------------------------------------------
def read_llms_json_to_list(bag):
    """ reads questions from a JSON file """
    file_path = "models_to_evaluate.json"
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            data = json.load(file)
            bag.list_of_llms = data.get("models_to_evaluate", None)  # Get the list or None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file: {file_path}")
        sys.exit(1)

#---------------------------------------------------------------
def string_to_list(list_of_chars):
    """ 
    # called from cmd_args_2()
    # Transforms list of chars to list of strings, divides by ',' 
    """
    ss = ''.join(list_of_chars)  # concatenate the list of chars into a single string
    ss = ss.lower()
    string_list = ss.split(',') 
    return string_list  # Output: ['abc', 'def', 'g']

#---------------------------------------------------------------
def check_for_correct_name(specified_list, list_of_sites):
    """ 
    # called from cmd_args_2()
    # checks if specified_list contains correct llm(s) names
    """
    return all(item in list_of_sites for item in specified_list)

#---------------------------------------------------------------
def cmd_args_1(bag):
    """ process cmd arguments """
    # -------------------------------------------------
    description_str = """ Script asks question and saves answers """
    bag.parser = argparse.ArgumentParser(description=description_str)
    # --------- specify llms (all | specific | exclude ) ----
    bag.parser.add_argument("-a", "--all", action='store_true',
        dest='all_llms', help="Ask questions to all llms")
    bag.parser.add_argument("-i", "--include", action='store',
        dest='include_llms', help="Ask questions to specific llm(s)")
    bag.parser.add_argument("-e", "--exclude", action='store',
        dest='exclude_llms', help="Ask questions to all llms exlude specific llm(s)")
    
    # ---------------------------------------------------------
    bag.parsed, unknown = bag.parser.parse_known_args()
    if len(unknown):
        print("unrecognized argument(s) " + str(unknown))
        sys.exit(1)

#---------------------------------------------------------------
def cmd_args_2(bag):
    """ args to include / exclude news sites """
    bag.llms = []

    if (not bag.parsed.all_llms 
        and not bag.parsed.include_llms 
        and not bag.parsed.exclude_llms):
        print("Error: need one of 3 options: -a -i -e, Exiting ...")
        print("Run with --help to see how to use the script")
        sys.exit(1)

    if bag.parsed.all_llms:
        bag.llms = bag.list_of_llms

    elif bag.parsed.include_llms:
        buff_include = string_to_list(bag.parsed.include_llms)
        if check_for_correct_name(buff_include, bag.list_of_llms):
            bag.llms = buff_include
        else:
            print("unrecognized llm(s) name: " + str(bag.parsed.include_news))
            sys.exit(1)

    elif bag.parsed.exclude_llms:
        buff_exclude = string_to_list(bag.parsed.exclude_llms)
        if check_for_correct_name(buff_exclude, bag.list_of_llms):
            bag.llms =  [x for x in bag.list_of_llms if x not in buff_exclude]            
        else:
            print("unrecognized llm(s) name: " + str(bag.parsed.exclude_news))
            sys.exit(1)

    if len(bag.questions_data['questions']) <= 0:
        print("Error - No questions to ask? Exiting ...")
        sys.exit(1)
        
    for llm in bag.llms:
        print(f"   llm = {llm}")

#---------------------------------------------------------------
def init_instruct_prompt_llm(bag):
    """ initialize instruct message for Ollama model """
    bag._ollama_instruct_message = bag.combined_question

#---------------------------------------------------------------
def init_ollama_model(bag):
    """ initialize Ollama model """
    bag.llm_ollama = ollama.Client()

#---------------------------------------------------------------
def write_results_to_json_file(bag):
    """ Write test results to JSON file """
    bag.script_dir = os.path.dirname(os.path.realpath(__file__))
    bag.dir_out = bag.script_dir + "/work/step2_answers"
    os.makedirs(bag.dir_out, exist_ok=True)
    str_clear1 = title_to_alphanum(bag.model_ollama)
    fname = (f"{bag.dir_out}/answers_{str_clear1}.json")
    with open(fname, "w", encoding="utf-8") as fh:
        print(f"writing {fname}")
        json.dump(bag.answers_results, fh, indent=4)

#---------------------------------------------------------------
def combine_question(bag):
    """ combine question and instruction into one string """
    bag.question     = bag.questions_data['questions'][bag.q_idx]['question']
    bag.instructions = bag.questions_data['questions'][bag.q_idx]['instructions']
    ss = f"""\n\n{bag.question}\n\n{bag.instructions}\n\n"""
    return ss

#---------------------------------------------------------------
def questions_part1(bag):
    """ part1 - called from create_questions() """
    bag.answers_results = { 
        "title"                  : "Answers of the llms",
        "llm_question_generator" : bag.questions_data['llm_question_generator'],
        "llm_answers"            : bag.model_ollama,
        "answers"                : [] 
    }
    print(f" ")
    print(f" ===================== Asking Questions Start ======================")
    print(f" === Answers for LLMS accuracy competition    =====================")
    print(f" === Questions generator: {bag.questions_data['llm_question_generator']} \n")
    print(f" === Asked models: {bag.llms}")

#---------------------------------------------------------------
def questions_part2_ask(bag):
    """ part2 - called from create_questions() in the loop, asks questions """

    init_instruct_prompt_llm(bag)
    print("Ask question")
    resp = bag.llm_ollama.generate(model=bag.model_ollama, prompt=bag._ollama_instruct_message)
    bag.asnwer_llm_instruct_prompt = resp['response']

#---------------------------------------------------------------
def questions_part3_print(bag):
    """ print results, called from create_questions() in the loop """
    print("\nAsked question:")
    print(f"\n{bag.combined_question}")
    print("\nAnswer:")
    print(f"\n{bag.asnwer_llm_instruct_prompt}")

#---------------------------------------------------------------
def create_questions(bag):
    """ Creates and asks questions """
    questions_part1(bag)
    
    for q_idx in range(len(bag.questions_data['questions'])):
        bag.q_idx = q_idx
        print(f" === Answers generating: {bag.q_idx} === model: {bag.model_ollama} ============")
        bag.combined_question = combine_question(bag)
        questions_part2_ask(bag) 
        questions_part3_print(bag)

        answer_info = {
            "question_id"                           : bag.q_idx,
            "asked_question"                        : bag.combined_question, 
            "answer_instruct_prompt_llm"            : bag.asnwer_llm_instruct_prompt,
        }
        
        bag.answers_results["answers"].append(answer_info)

# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------
bag = MyBunch()
read_questions_from_json_file(bag)
read_llms_json_to_list(bag)
cmd_args_1(bag)
cmd_args_2(bag)
init_ollama_model(bag)
for model in bag.llms:
    bag.model_ollama = model
    create_questions(bag)
    write_results_to_json_file(bag)
