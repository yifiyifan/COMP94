"""
    1. Solution to build job family * level of experience grid 
    2. Use sentence encoder to find top x job posting by title
"""

from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from itertools import product
import string

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def initialise_job_exp_grid(job_family_list:list, level_of_experience_list:list):
    grid_lst = list(product(job_family_list, level_of_experience_list))
    grid_df = pd.DataFrame(
        data=grid_lst,
        columns=["job_family","level_of_experience"]
    )
    return grid_df

def validate_output(answer:str, choices:dict):
    return answer.upper() in choices.keys()

def query_flan_t5(
    model:T5ForConditionalGeneration,
    tokenizer:T5Tokenizer,
    context:str,
    question:str,
    choices:dict = None, # mapping of options [A-Z]{1} -> some text
    return_int:bool = False,
):
    input_text = f"question: {question} context: {context}"
    if choices is not None:
        choices_txt = " ".join([
            key + ") " + val
            for key, val in choices.items()
        ])
        input_text += f" choices: {' '.join(choices_txt)}"
    inputs = tokenizer(input_text, return_tensors='pt')

    # Generate the answer
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if choices is not None:                     # if multiple choice
        if validate_output(answer, choices):    # check if predicted option is one of the available options
            return choices[answer.upper()]      # return the value of predicted option (expect flan-t5 to predict just A for option A)
        else:
            return None 
    else:
        if return_int:
            try:
                return str(int(answer))
            except:
                raise TypeError("Requested to return integer. prediction cannot be converted ot integer")
        return answer                           # if not multiple choice, return the predicted value directly

def build_choices_dict(options_list:list[str], incl_others:bool=True):
    if incl_others:
        options_list.append("none of the above")
    num_options = len(options_list)
    return {key:val for key, val in zip(string.ascii_uppercase[0:num_options], options_list)}

def extract_job_attribute(
    job_post_text:str, 
    job_family_options:list[str], 
    level_of_experience_options:list[str],
    model:T5ForConditionalGeneration,
    tokenizer:T5Tokenizer,
):
    """return job family, level of experience, minimum years of experience"""
    
    q1_question = "What is the job family described in this job post?"
    q1_choices = build_choices_dict(job_family_options)
    q1_answer = query_flan_t5(model, tokenizer, job_post_text, q1_question, q1_choices)

    q2_question = "What is the minimum number of years of experience required in this job post answer as an integer?"
    q2_answer = query_flan_t5(model, tokenizer, job_post_text, q2_question, return_int=True)

    q3_question = "What level of experience is described in this job posting?"
    q3_choices = build_choices_dict(level_of_experience_options)
    q3_answer = query_flan_t5(model, tokenizer, job_post_text, q3_question, q3_choices)

    return q1_answer, q2_answer, q3_answer

def find_job_posting(job_family:str, level_of_experience:str, job_posting_df) -> list:
    """Return a list of job_id"""
    keyword = level_of_experience + " " + job_family
    pass

