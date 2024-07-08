import re 
from itertools import product
import pandas as pd
from typing import (List, Tuple)

def clean_string(text):
    # Define the regex pattern to match all characters except alphanumeric and regular punctuations
    pattern = r'[^a-zA-Z0-9.,!?;:\'\"()\[\]{}<>@#$%^&*+=\-_~`\s]'
    # Substitute all characters matching the pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def format_execution_time(start_time, end_time):
    elapsed_time = end_time - start_time

    # Format the elapsed time into hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    time_text = f"{seconds:.2f} seconds"
    if minutes > 0:
        time_text = f"{int(minutes)} minute " + time_text
    if hours > 0:
        time_text = f"{int(hours)} hours " + time_text
    # Print the formatted elapsed time
    return time_text

def initialise_job_exp_grid(job_family_list:List, experience_list:List[Tuple]):
    df_job_fam = pd.DataFrame({"job_family":job_family_list})
    df_exp = pd.DataFrame({"exp_range":experience_list})
    df_exp[['exp_lower', 'exp_upper']] = pd.DataFrame(df_exp["exp_range"].tolist(), index=df_exp.index)
    df_output = df_job_fam.join(df_exp, how='cross')[["job_family", "exp_lower", "exp_upper"]]
    return df_output
    # grid_lst = list(product(job_family_list, level_of_experience_list))
    # grid_df = pd.DataFrame(
    #     data=grid_lst,
    #     columns=["job_family","experience_range"]
    # )
    # grid_df = grid_df.assign()
    # return grid_df