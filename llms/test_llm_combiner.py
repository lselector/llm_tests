
"""
# This script uses LLM to combine summarized news for multiple (web) sites.
# For each sites it reads one JSON file with summaries
# from directory news/work/step1_summaries/
# 
# You can use local or API LLMs (ollama, OpenAI, groq, etc.)
# Usage:
# python llm_news_sumarizer.py -h   # show help, list cmd options
# default model - ollama llama3-gradient
# -aa - use OpenAI LLM, 
# -s ... -m ... # specify LLM source and model
"""

import os, sys, argparse, json, time, glob
from string import Template
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from openai import OpenAI

from mybag import *
from myutils import *

#---------------------------------------------------------------
def get_news_files(bag):
    """ make list of input paths under news/work/step1_summaries """
    if test_avail(bag, 'news_files'):
        return bag.news_files
    bag.buffpath = f"/home/{bag.USER}/GitHub/aiwork/scrapers" + "/news/work/step1_summaries"
    news_files = []
    print(f"\nDo search in folder {bag.buffpath}")
    print("\nThe news summarized files found:\n")
    i = 0
    for entry in os.scandir(bag.buffpath):
        if entry.is_file():
            news_files += [entry.name]
            print(f"{i + 1}. {news_files[i]}")
            i+=1
    bag.news_files = news_files

#---------------------------------------------------------------
def cmd_args_1(bag):
    """ define bag.parser arguments for CMD arguments """
    # -------------------------------------------------
    description_str = """
        This script combines news from multiple JSON files under news/work/step1_summaries.
        It removes duplicates and ranks news for CRO.
        Then outputs final JSON file.
    """
    bag.parser = argparse.ArgumentParser(description=description_str)
    # --------- specify LLM (default llama3 --------------------
    bag.parser.add_argument("-aa", action='store_true', default=False, dest='arg_a',
        help="Use OpenAI service with gpt-4o model (overrides -s and -m)" )
    bag.parser.add_argument("-mm", action='store_true', default=False, dest='arg_mm',
        help="Use Ollama Mistral model (overrides -s and -m)" )
    bag.parser.add_argument("-s", default="ollama", action='store',
        dest='arg_service', help="LLM service provider (default: ollama)")
    bag.parser.add_argument("-m", default="llama3-gradient", action='store',
        dest='arg_model', help="Specific LLM model (default: llama3-gradient)")

    bag.parsed, unknown = bag.parser.parse_known_args()
    print(f"PARSED: {bag.parsed}")
    if len(unknown):
        print("unrecognized argument(s) " + str(unknown))
        sys.exit(1)
    if len(bag.news_files) <= 0:
        print("Error - No news to parse? Exiting ...")
        sys.exit(1)
    for sum_file in bag.news_files:
        print(f"   sum_file = {sum_file}")

# --------------------------------------------------------------
def cmd_args_2(bag):
    """ args for service and model """
    bag.arg_service = "ollama"
    bag.arg_model = "llama3-gradient"

    if bag.parsed.arg_service:
        bag.arg_service = bag.parsed.arg_service
    if bag.parsed.arg_model:
        bag.arg_model = bag.parsed.arg_model

    bag.arg_a = False
    if bag.parsed.arg_a:
        bag.arg_service = "openai"
        bag.arg_model = "gpt-4o"
    
    bag.arg_mm = False
    if bag.parsed.arg_mm:
        bag.arg_model = "mistral"

    print("arg_service = ", bag.arg_service)
    print("arg_model   = ", bag.arg_model)

# --------------------------------------------------------------
def process_cmd_args(bag):
    """
    # processes cmd arguments - and populates bag.arg_* elements
    # specify model and news-sources - 
    # see in define_cmd_args(bag)
    """
    cmd_args_1(bag)
    cmd_args_2(bag)

#---------------------------------------------------------------
def ollama_init(bag):
    """ initialize Ollama model, called from select_model(bag) """
    model = Ollama(model=bag.arg_model, num_ctx=65536)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You're a helpful assitant.",),
         ("human", "{question}"),]
    )
    bag.runnable = prompt | model | StrOutputParser()

#---------------------------------------------------------------
def openai_init(bag):
    """ initialize OpenAI model, called from select_model(bag) """
    llm = ChatOpenAI(model=bag.arg_model, temperature=0, max_tokens=1000)
    template = """You're a helpful assitant.\n\n{question}\n"""
    prompt = PromptTemplate.from_template(template)
    bag.runnable = prompt | llm
    
    bag.model = OpenAI()


# --------------------------------------------------------------
def select_model(bag):
    """ checking input arguments """
    bag.runnable = None
    bag.model_type = bag.arg_service + "--" + bag.arg_model
    if bag.arg_service == "ollama":
        ollama_init(bag)
    elif bag.arg_service == "openai":
        openai_init(bag)
    else:
        print("wrong model service ", bag.arg_service, "\nExiting...")
        sys.exit(1)

#---------------------------------------------------------------
def read_json_file(bag):
    """ reads a JSON file """
    with open(bag.file_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    sub_news_list = data['sub_news']
    for sub_news_item in sub_news_list:
        extracted_data = {
            'id'                 : bag.news_id,
            'sub_title'          : sub_news_item['sub_title'],
            'sub_news_content'   : ' '.join(sub_news_item['sub_news_content']),  # Combine into a single string
            'link'               : sub_news_item['link'],
            'date'               : sub_news_item['date'],
            'date_run'           : sub_news_item['date_run'],
            'enforcement_action' : sub_news_item['enforcement_action'],
            'tags'               : sub_news_item['tags']
        }
        bag.combined_news.append(extracted_data)
        bag.news_id_list.append(bag.news_id)
        bag.news_id+=1

#---------------------------------------------------------------
def combine_json_news(bag):
    """ 
    # Combines news dicts from different summarized JSON files
    # into list bag.combined_news (list of dicts)
    """
    print(f"\nCombining news from JSON files:\n")
    bag.combined_news = []
    bag.news_id       = 0
    bag.news_id_list  = []
    for sum_file in bag.news_files:
        bag.file_path = f"/home/{bag.USER}/GitHub/aiwork/scrapers" + f"/news/work/step1_summaries/{sum_file}"
        print(f"reading {bag.file_path}")
        read_json_file(bag)
    N = len(bag.combined_news)
    print(f"Finished loading {N} news into list bag.combined_news")
        
#---------------------------------------------------------------
def create_prompt_dedup_2_tasks_openai(bag): 
    """ creates a request for LLM to remove duplicates for combined news
        # 1 task - create a comma-separated duplates news list
        # 2 task - create a comma-separated list without duplicate news
    """ 
    print("\ncreate_prompt_dedup\n")
    
    news_str = str(bag.combined_news)
    tt = Template("""
        You are very serious news duplicates detector.
        You are such a serious that you print only answer, nothing else.
                  
        Your main tast is to create the list without duplicate news

        You have a list of firm rules that you always follow:
            1. Analyze the symantic and structure of each news inside the list of news.
               You know that each news structure contains a piece of:
               id, Title, News, Link, Date, Date_run, Tags. You analyze each peace of news structure
            2. You compare each analyzed news to other news in the news list.
               There always will be duplicates in the list of news and you always detect them.
               There can be more then one duplicate of each news.
               If you see two or more duplicates of the same news, you keep only one first news that are unic, other news you put into duplates news list
            3. After detecting duplicates you put them into a comma-separated duplates news list of numeric values of "id" fields.
               You always use this example as pattern for duplates news list: [1,2,3,4,5]
            4. You create a comma-separated list without duplicate news. How you do it:
                   You take the list of news and delete all the ids from list of news if they are in duplates news list
            5. You print list without duplicate news as an answer.
                  
        List of news:
        \n News:\n$news\n""")
    #bag.prompt = tt.substitute(news=news_str)

    bag.messages_config = [
            {"role": "system", "content": tt.substitute(news=news_str)},
        ]
    
#---------------------------------------------------------------
def create_prompt_dedup_1_task_openai(bag): 
    """ creates a request for LLM to remove duplicates for combined news """
    print("\ncreate_prompt_dedup\n")
    
    news_str = str(bag.combined_news)
    tt = Template("""
        You are very serious news duplicates detector.
        You are such a serious that you print only answer, nothing else.
                  
        Your main tast is to create a duplates news list.

        You have a list of firm rules that you always follow:
            1. Analyze the symantic and structure of each news inside the list of news.
               You know that each news structure contains a piece of:
               id, Title, News, Link, Date, Date_run, Tags. You analyze each peace of news structure
            2. You compare each analyzed news to other news in the news list.
               There always will be duplicates in the list of news and you always detect them
               There can be more then one duplicate of each news.
               If you see two or more duplicates of the same news, you keep only one first news that are unic, other news you put into duplates news list
            3. After detecting duplicates you put them into a comma-separated duplates news list of numeric values of "id" fields.
               You always use this example as pattern for duplates news list: [1,2,3,4,5]
                  
        List of news:
        \n News:\n$news\n""")
    #bag.prompt = tt.substitute(news=news_str)

    bag.messages_config = [
            {"role": "system", "content": tt.substitute(news=news_str)},
        ]

#---------------------------------------------------------------
def is_int(s):
    """ check if the string represents an integer """
    try:
        int(s)
        return True
    except ValueError:
        return False

#---------------------------------------------------------------
def remove_duplicates(bag):
    """
    # use prompt to deduplicate 
    # (in future may use chroma_db or other vector db)
    """
    print("\nRemoving duplicates\n")

    response = bag.model.chat.completions.create(
                model=bag.arg_model,
                messages=bag.messages_config,
                max_tokens=4000,
                temperature=0.5
            )

    ss = response.choices[0].message.content

    print(f"raw responce: {ss}")

    N_before = len(bag.combined_news)
    #ss = bag.runnable.invoke(bag.prompt, config=RunnableConfig(callbacks=[]))
    if type(ss) != str:
        ss = ss.content
#    print("-"*100,f"\n{ss}\n","-"*100)
    ss = re.sub(r"[^\d,]", " ", ss)  # keep only numbers and commas
    ss = re.sub(r"\s", "", ss)       # remove spaces
    bag.ss = ss
    mylist = ss.split(',')
    mylist = [x.strip() for x in mylist]
    bag.mylist = mylist
    mylist2 = []
    for x in mylist:
        if is_int(x):
            mylist2.append(int(x))
    mylist2 = list(set(mylist2)) # remove dups from output
    mylist2.sort(key=int)        # in-place numeric sorting
    N_after = len(mylist2)
    print(f"Before/After : {N_before}/{N_after}")
    print(f"\nList of deduped news id-s: {mylist2}")
    N_max = N_before -1
    while len(mylist2) > 0 and mylist2[-1] > N_max:
        print("Removing extra elements in mylist2\n")
        mylist2 = mylist2[:-1]
    
    N_after = len(mylist2)
    print(f"Adjusted mylist2 len: {N_after}")
    print("Renumbering the dedupped news")
    bag.combined_news_no_dups = []
    new_id = 0
    bag.mylist2 = mylist2      # put it in bag for debugging
    for id in bag.mylist2:
        bag.id = id            # put it in bag for debugging
        bag.new_id = new_id    # put it in bag for debugging
        bag.combined_news[id]['id'] = new_id
        bag.combined_news_no_dups.append(bag.combined_news[id])
        new_id += 1
    N_final = len(bag.combined_news_no_dups)
    print(f"final length of bag.combined_news_no_dups = {N_final}")

#---------------------------------------------------------------
def read_profile(bag):
    """ read prfile for CRO into bag.profile_str """
    fname = f"/home/{bag.USER}/GitHub/aiwork/scrapers" + "/profiles/ChiefRiskOfficer.txt"
    print(f"\nreading profile {fname}\n")
    with open(fname, "r", encoding="utf-8") as fh:
        bag.profile_str = fh.read()

#---------------------------------------------------------------
def create_prompt_rank_openai(bag): 
    """ 
    # creates bag.prompt for LLM to rank (sort) news 
    # in order of importance for Chief Risk Officer (CRO) 
    """
    print("\ncreate_prompt_rank\n")
    bag.N_news = len(bag.combined_news_no_dups)
    news_str = str(bag.combined_news_no_dups)

    tt = Template("""
        You are very serious Chief Risk Officer news ranker.
        You are such a serious that you print only answer, nothing else.
                  
        Your main task is ranked list of news for Chief Risk Officer
        You never miss any news in list of news, list of news has $count news. Length of ranked list of news must be the same as list of news.
                  
        You know Chief Risk Officer very well but you always see at the points you noted about Chief Risk Officer:
        $desc
                  
        You have a list of firm rules that you always follow:
            1. Analyze the symantic and structure of each news inside the list of news.
               You know that each news structure contains a piece of:
               id, Title, News, Link, Date, Date_run, enforcement_action, Tags. You analyze each peace of news structure
               Especialy you see at enforcement_action. If enforcement_action = 1 you will always put it at the top of the rating list.
               If you see compared news not neccesary for Chief Risk Officer at all you never delete it you just put it to the botoom of the rating list.
            2. You compare each analyzed news to other news in the news list.
               If you one news you compare more relevant for Chief Risk Officer to second news you put it at the higher at ranked list of news and second news lower at the ranked list of news
               You always use this example as pattern for ranked list of news: [1,2,3,4,5]
            3. Print the ranked list of news as answer.
        
      """)
    
    bag.messages_config = [
            {"role": "system", "content": tt.substitute(desc=bag.profile_str,
                               news=news_str, count=bag.N_news)},
        ]
    
#---------------------------------------------------------------
def create_prompt_rank(bag): 
    """ 
    # creates bag.prompt for LLM to rank (sort) news 
    # in order of importance for Chief Risk Officer (CRO) 
    """
    print("\ncreate_prompt_rank\n")
    bag.N_news = len(bag.combined_news_no_dups)
    news_str = str(bag.combined_news_no_dups)

    tt = Template("""
        Please reset the context and start fresh.
        Please do not consider previous questions in this session.
        
        Below I provide a list of JSON structures containing news items.
        Each structure consisting of the following fields:

            "id",
            "sub_title",
            "sub_news_content",
            "link",
            "date",
            "date_run",
            "enforcement_action",
            "tags".
        
        I will also provide a description of a 
        manager called CRO (Chief Risk Officer).
        
        Please reorder these structures in order of importance
        for a Chief Risk Officer according to provided description.
        
        The news which are risk related should go first.
        
        Please take the values of "id" fields in the 
        in the new re-sorted order and print them out.
        The output list should contain the same number of items
        as the nubmer of news items.

        Print only this re-sorted list of id-s in the new order.
        Do not print anything before or after the list.

        Here is the list of structures to process:
        \n\n News:\n$news\n
                  
        And here is the description of Chief Risk Officer:
        \n\nProfile:\n$desc\n""")

    bag.prompt = tt.substitute(desc=bag.profile_str,
                               news=news_str, count=bag.N_news)

#---------------------------------------------------------------
def rank_for_CRO(bag):
    """ use LLM to rank for CRO """
    print("\nre-ranking for CRO\n")
    N_before = len(bag.combined_news_no_dups)
    ss = bag.runnable.invoke(bag.prompt, config=RunnableConfig(callbacks=[]))
    if type(ss) != str:
        ss = ss.content
#    print("-"*100,f"\n{ss}\n","-"*100)
    ss = re.sub(r"[^\d,]", " ", ss)  # keep only numbers and commas
    ss = re.sub(r"\s", "", ss)       # remove spaces
    mylist = ss.split(',')
    mylist = [x.strip() for x in mylist]
    mylist2 = []
    for x in mylist:
        if is_int(x):
            mylist2.append(int(x))
    N_max = N_before -1
    mylist3 = []
    for ii in range(len(mylist2)):
        if 0 <= mylist2[ii] <= N_max:
            mylist3.append(mylist2[ii])
    N_after = len(mylist3)
    print(f"Before/After : {N_before}/{N_after}")
    print(f"\nList of returned reordered news id-s: {mylist3}")
    print("Renumbering the re-ranked news")
    s1 = set(list(range(N_before))) # set of original id-s
    s2 = set(mylist3)               # subset of selected id-s
    s3 = s1-s2                      # remaining id-s
    bag.combined_news_reranked = []
    new_id = 0
    for id in mylist3:
        bag.combined_news_reranked.append(bag.combined_news_no_dups[id])  # add to list
        bag.combined_news_reranked[new_id]['id'] = new_id                 # give it a number
        new_id += 1
    for id in sorted(s3): # remaining news
        bag.combined_news_reranked.append(bag.combined_news_no_dups[id])  # add to list
        bag.combined_news_reranked[new_id]['id'] = new_id                 # give it a number
        new_id += 1

    print(f"len(bag.combined_news_reranked) = {len(bag.combined_news_reranked)}")

#---------------------------------------------------------------
def save_to_JSON(bag):
    """
    # save dedupped reranked news into JSON file under news/work/step2_combined/
    # with date in file name
    """
    bag.dir_out = bag.script_dir + "/news/work/step2_combined"
    os.makedirs(bag.dir_out, exist_ok=True)
    fname = f"{bag.dir_out}/combined.json"
    print(f"\nwriting {fname}\n")
    final_news = { "title": "Title of the news", "sub_news": bag.combined_news_reranked }
    with open(fname, "w", encoding="utf-8") as fh:
        json.dump(final_news, fh, indent=4)

# --------------------------------------------------------------
def bag_init(bag):
    """ adds script name and directory to bag """
    full_name  = sys.argv[0]
    short_name = full_name.split('/')[-1]
    bag.full_script_name  = full_name
    bag.short_script_name = short_name
    bag.script_cmd        = "python %s" % (' '.join(sys.argv))
    bag.script_start_time = time.time()
    bag.script_dir = os.path.dirname(os.path.realpath(__file__))

def print_elapsed_time(bag, t1=None):
    """
    # prints current date/time, and time elapsed from beginning of the script
    # optionally accepts another starting point (in epoch seconds)
    # depends on bag.script_start_time and bag.script_cmd
    """
    time_now   = now_str()
    if not t1:
        t1 = bag.script_start_time
    elapsed    = elapsed_time_hms(bag, t1)
    print(f"{time_now} FINISHED, elapsed time was {elapsed}")

def print_elapsed_time_for_model(bag, t1=None):
    """
    # prints current date/time, and time elapsed from beginning of the script
    # optionally accepts another starting point (in epoch seconds)
    # depends on bag.script_start_time and bag.script_cmd
    """
    time_now   = now_str()
    t1 = bag.t1
    t2 = time.time()
    elapsed    = sec_to_hms(round(t2-t1,2))
    
    print(f"{time_now} FINISHED, elapsed time was {elapsed}")


# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------

bag = MyBunch()
bag.USER = os.environ['USERNAME']
bag_init(bag)
get_news_files(bag)
process_cmd_args(bag)
combine_json_news(bag)
print("\n\n\n\n\n")
print("\n\n =================================== START TESTING =================================")

bag.t1 = time.time()
select_model(bag) # set bag.runnable
print(f"\n\n === Testing model: {bag.arg_model} === Testing prompt: 1 task at a time ====================\n")
print(f" === output data: ==================================================================\n")
create_prompt_dedup_1_task_openai(bag)  # bag.prompt
remove_duplicates(bag)
print(f"\n ====================================================================================\n")
#read_profile(bag)
#create_prompt_rank(bag)
#rank_for_CRO(bag)
#save_to_JSON(bag)
print(" === Time of task doing ============================================================= ")
print_elapsed_time_for_model(bag)

print(f"\n ==Finished testing model: {bag.arg_model}====================================================\n")

bag.t1 = time.time()
select_model(bag) # set bag.runnable
print(f"\n\n === Testing model: {bag.arg_model} === Testing prompt: 2 tasks at a time ====================\n")
print(f" === output data: ===================================================================\n")
create_prompt_dedup_2_tasks_openai(bag)  # bag.prompt
remove_duplicates(bag)
print(f"\n ===================================================================================\n")
#read_profile(bag)
#create_prompt_rank(bag)
#rank_for_CRO(bag)
#save_to_JSON(bag)
print(" === Time of task doing ============================================================= ")
print_elapsed_time_for_model(bag)

print(f"\n === Finished testing model: {bag.arg_model} ==================================================\n")

print(" \n\n=== Time of testing ============================================================= ")
print_elapsed_time(bag)
print("\nDONE\n")
