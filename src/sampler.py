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

def validate_output(answer:str, choices:dict|list):
    if isinstance(choices, list):
        return answer in choices
    else:
        return answer in choices.keys()

def build_input_text(
    question:str,
    context:str = None,
    choices:dict | list = None, # mapping of options [A-Z]{1} -> some text
):
    output = f"question: {question}"

    if choices is not None:
        if isinstance(choices, list):
            choices_txt = ", ".join(choices)
        else:
            choices_txt = " ".join([
                f"{key}) {val}"
                for key, val in choices.items()
            ])
        output += f" choices: {choices_txt}"
        # return f"question: {question} choices: {choices_txt} context: {context}"
    if context is not None:
        output += f" context: {context}"
    
    return output

def query_flan_t5(
    model:T5ForConditionalGeneration,
    tokenizer:T5Tokenizer,
    question:str,
    context:str = None,
    choices:dict | list = None, # mapping of options [A-Z]{1} -> some text
    return_int:bool = False,
    num_try:int=0,
    max_try:int=3,
    max_new_tokens:int=50
):

    input_text = build_input_text(question=question, context=context, choices=choices)
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).to(model.device)

    # Generate the answer
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if choices is not None:                     # if multiple choice
        if validate_output(answer, choices): 
            if isinstance(choices, list):
                return answer.upper()
            else:  
                return choices[answer.upper()]      # return the value of predicted option (expect flan-t5 to predict just A for option A)
        else:
            if num_try < max_try:
                conversion_question = f"Please choose a single letter to represent the context."
                if isinstance(choices, list):
                    conversion_choices = choices 
                else:
                    conversion_choices = list(choices.keys())
                cleaned_answer:str = query_flan_t5(
                    model, tokenizer, question=conversion_question, context=answer, choices=conversion_choices, return_int=False, num_try=num_try+1
                )
                if num_try == 0: # initial layer
                    return choices[cleaned_answer.upper()]
                else:
                    return cleaned_answer.upper()
            else:
                return None 
    else:
        if answer.lower() == "not specified":
            return None
        if return_int:
            try:
                return str(int(answer))
            except:
                if num_try < max_try:
                    conversion_question = "Please return a single integer to represent the number range in context. If the context suggest it is not specified, say 'not specified'"
                    return query_flan_t5(
                        model, tokenizer, question=conversion_question, context=answer, return_int=True, num_try=num_try+1
                    )
                else:
                    return None
        
        return answer                           # if not multiple choice, return the predicted value directly

def chunk_context(question:str, context:str, overlap:int, tokenizer:T5Tokenizer, choices:dict|list=None, buffer:int=5) -> list:
    input_template = build_input_text(question=question, context="placeholder", choices=choices)
    tokens_excl_context = tokenizer.encode(input_template)
    context_max_len = 512-len(tokens_excl_context)-buffer
    
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    chunks = []
    
    start = 0
    while start < len(context_tokens):
        end = min(start + context_max_len, len(context_tokens))
        chunk = context_tokens[start:end]
        chunks.append(chunk)
        if end == len(context_tokens):
            break
        start = end - overlap  # Create overlap
    return [tokenizer.decode(c) for c in chunks]

def summarize_chunk_answers(model:T5ForConditionalGeneration, tokenizer:T5Tokenizer,chunks:dict, aggregate:bool, question:str):
    if aggregate:
        # combine all answers to produce an answer
        query = f"answer the question '{question}' by combining and summarizing the context"
        return query_flan_t5(model, tokenizer, query, context=", ".join(chunks))
    else:
        # choose one that best 
        query = f"answer the question '{question}' by choosing one of the choices"
        return query_flan_t5(model, tokenizer, query, choices=chunks)
 

def build_choices_dict(options_list:list[str], incl_others:bool=True):
    tmp_list = options_list.copy() # make a copy to not mess with the original list
    if incl_others:
        tmp_list.append("none of the above")
    num_options = len(tmp_list)
    return {key:val for key, val in zip(string.ascii_uppercase[0:num_options], tmp_list)}


def extract_job_title(
    job_post_text:str, 
    job_family_options:list[str], 
    model:T5ForConditionalGeneration,
    tokenizer:T5Tokenizer,
):
    """return job family, level of experience, minimum years of experience"""

    q1_question = "What is the job family described in the job title and job description?"
    q1_choices = build_choices_dict(job_family_options)
    q1_answer = query_flan_t5(model, tokenizer, q1_question, job_post_text, q1_choices)

    return q1_answer

def extract_years_of_experience(
    job_post_text:str,  
    model:T5ForConditionalGeneration,
    tokenizer:T5Tokenizer,
    chunk_overlap:int=20,
):
    question = "What is the minimum number of years of experience specified in the context as an integer? It is possible that the context did not mention a minimum years of experience."
    context_chunks = chunk_context(question, job_post_text, chunk_overlap, tokenizer)

    chunk_answer = [query_flan_t5(model, tokenizer, question, c, return_int=True) for c in context_chunks]
    chunk_answer = [a for a in chunk_answer if a is not None]

    if len(chunk_answer) > 1:
        summ_question = "Which choice has the smallest integer?"
        choices_dict = build_choices_dict(chunk_answer, incl_others=False)
        tmp_output = summarize_chunk_answers(model, tokenizer, choices_dict, aggregate=False, question=summ_question)
    if len(chunk_answer) == 1:
        tmp_output = chunk_answer[0]
    else:
        return None
    
    hallucination_check_q = "Does the context suggest applicant require at least {tmp_output} years of experience"

    if hallucination_check(
        job_post_text,
        hallucination_check_q,
        model,
        tokenizer
    ):
        return tmp_output
    else:
        return None

def extract_required_skills(
    job_post_text:str,  
    job_family:str,
    model:T5ForConditionalGeneration,
    tokenizer:T5Tokenizer,
    chunk_overlap:int=20,    
):
    question = "What skills does the job applicant must have for this role? do not include qualification and years of experience."
    context_chunks = chunk_context(question, job_post_text, chunk_overlap, tokenizer)

    chunk_answer = [query_flan_t5(model, tokenizer, question, c, max_new_tokens=100) for c in context_chunks]
    chunk_answer = [a for a in chunk_answer if a is not None]

    if len(chunk_answer) > 1:
        summ_question = f"Deduplicate and summarise the context to describe the skillsets required for a {job_family} role?"
        choices_dict = build_choices_dict(chunk_answer, incl_others=False)
        return summarize_chunk_answers(model, tokenizer, chunk_answer, aggregate=True, question=summ_question)
    elif len(chunk_answer) == 1:
        return chunk_answer[0]
    else:
        return None
    
def hallucination_check(
    job_post_text:str,  
    hallucination_check_q:str, # question that if any chunk responded ture, return true
    model:T5ForConditionalGeneration,
    tokenizer:T5Tokenizer,
    chunk_overlap:int=20,
) -> bool:
    
    context_chunks = chunk_context(hallucination_check_q, job_post_text, chunk_overlap, tokenizer)
    choices = build_choices_dict(["True", "False"], incl_others=False) 
    chunk_answer = [query_flan_t5(model, tokenizer, hallucination_check_q, c, choices) for c in context_chunks]
    chunk_answer = [bool(a) for a in chunk_answer if a is not None]

    if len(chunk_answer) > 0:
        return any(chunk_answer)
    else:
        return False
    
