# from src.scrap_data import (
#     scrape_resume_link,
#     get_resume_info,
# )

from src.sampler import (
    extract_job_attribute,
    extract_job_title,
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

JOB_POSTING_PATH = os.path.join(os.getcwd(), "data", "job_postings_dev.csv")
FLAN_T5_MODEL_NAME = "google/flan-t5-large"

if __name__ == "__main__":
    
    job_posting_df = pd.read_csv(JOB_POSTING_PATH)

    tokenizer = T5Tokenizer.from_pretrained(FLAN_T5_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(FLAN_T5_MODEL_NAME)

    job_id_col = job_posting_df["job_id"]
    job_postings_desc_col = job_posting_df["description"]
    job_title_col = job_posting_df["title"]
    
    # job_family_col = []
    # level_of_experience_col = []
    # years_of_experience_col = []
    # required_skills_col = []

    extracted_info = []

    for id, title, desc in tqdm(zip(job_id_col, job_title_col, job_postings_desc_col)):
        job_fam = extract_job_title(
            job_post_text= f"job title in this job post is {title}",
            job_family_options=CONFIG["job_family"],
            model=model,
            tokenizer=tokenizer
        )

        if job_fam in CONFIG["job_family"]:
            # TODO: add confidence scoring to filter out bad answers
            years, level, skill = extract_job_attribute(
                job_post_text= f"job title: {title}. description: {desc}",
                level_of_experience_options=CONFIG["experience_level"],
                definitions_loc=CONFIG["definitions_loc"],
                model=model,
                tokenizer=tokenizer
            )

            extracted_info.append((id, title, desc, job_fam, years, level, skill))
        else:
            extracted_info.append((id, title, desc, "out of scope", None, None, None))

    job_posting_df_extracted = pd.DataFrame(
        data=extracted_info,
        columns=["id", "title", "desc", "job_fam", "years", "level", "skill"]
    )

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

