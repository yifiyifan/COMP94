from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from src.gpt_sampler import ResumeMatcher
from src.sampler import clean_df

if __name__ == "__main__":

    # \data\example_input\jd-5-holdout-examples.csv
    job_post_df = pd.read_csv(os.path.join("data", "example_input", "jd-5-holdout-examples.csv"))[["id","title","desc"]]
    res_df = pd.read_csv(os.path.join("data", "example_input", "resume-10-holdout-examples.csv"))[["resume"]]

    pairwise = job_post_df.join(
        res_df,
        how="cross"
    )
    
    pairwise.to_csv(os.path.join("data", "example_input", "holdout-examples.csv"), index=False)