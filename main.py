# from src.scrap_data import (
#     scrape_resume_link,
#     get_resume_info,
# )

from src.sampler import (
    extract_job_attribute,
)


import os 
import yaml
import pandas as pd
from tqdm import tqdm
from datetime import datetime


from transformers import T5Tokenizer, T5ForConditionalGeneration


CONFIG = yaml.load(
    open(os.path.join(os.getcwd(), "config.yaml"), "r+"),
    Loader = yaml.FullLoader
)

JOB_POSTING_PATH = os.path.join(os.getcwd(), "data", "job_postings_dev_debug.csv")
FLAN_T5_MODEL_NAME = "google/flan-t5-large"

if __name__ == "__main__":
    
    job_posting_df = pd.read_csv(JOB_POSTING_PATH)

    tokenizer = T5Tokenizer.from_pretrained(FLAN_T5_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(FLAN_T5_MODEL_NAME)

    job_postings_desc_col = job_posting_df["description"]
    job_title_col = job_posting_df["title"]
    
    job_family_col = []
    level_of_experience_col = []
    years_of_experience_col = []
    required_skills_col = []

    for title, desc in tqdm(zip(job_title_col, job_postings_desc_col)):
        # TODO: add confidence scoring to filter out bad answers
        job_fam, years, level, skill = extract_job_attribute(
            job_post_text= f"job title: {title}. description: {desc}",
            job_family_options=CONFIG["job_family"],
            level_of_experience_options=CONFIG["experience_level"],
            definitions_loc=CONFIG["definitions_loc"],
            model=model,
            tokenizer=tokenizer
        )
        job_family_col.append(job_fam)
        level_of_experience_col.append(level)
        years_of_experience_col.append(years)
        required_skills_col.append(skill)

    job_posting_df_extracted = job_posting_df[["job_id", "title", "description"]]
    job_posting_df_extracted["job_family"] = job_family_col
    job_posting_df_extracted["level"] = level_of_experience_col
    job_posting_df_extracted["years_of_experience"] = years_of_experience_col
    job_posting_df_extracted["skills"] = required_skills_col

    try:
        job_posting_df_extracted.to_csv(
            os.path.join(os.getcwd(), "data", "job_posting_transformed.csv"),
            index=False
        )
    except PermissionError:
        time_stamp = current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        job_posting_df_extracted.to_csv(
            os.path.join(os.getcwd(), "data", f"job_posting_transformed_{time_stamp}.csv"),
            index=False
        )
