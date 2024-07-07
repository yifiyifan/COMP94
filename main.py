# from src.scrap_data import (
#     scrape_resume_link,
#     get_resume_info,
# )

from src.sampler import (
    extract_job_title,
    extract_required_skills,
    extract_years_of_experience,
    clean_string,
)


import os 
import yaml
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration


CONFIG = yaml.load(
    open(os.path.join(os.getcwd(), "config.yaml"), "r+"),
    Loader = yaml.FullLoader
)

JOB_POSTING_PATH = os.path.join(os.getcwd(), "data", "job_postings_dev_debug.csv")
FLAN_T5_MODEL_NAME = "google/flan-t5-large"

if __name__ == "__main__":
    
    job_posting_df = pd.read_csv(JOB_POSTING_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained(FLAN_T5_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(FLAN_T5_MODEL_NAME).to(device)

    job_id_col = job_posting_df["job_id"]
    job_postings_desc_col = job_posting_df["description"]
    job_title_col = job_posting_df["title"]
    
    extracted_info = []

    for id, title, desc in zip(job_id_col, job_title_col, job_postings_desc_col):
        try:
            job_fam = extract_job_title(
                job_post_text= f"job title in this job post is {title}",
                job_family_options=CONFIG["job_family"],
                model=model,
                tokenizer=tokenizer
            )
            clean_desc = clean_string(desc)
            if job_fam in CONFIG["job_family"]:
                years = extract_years_of_experience(
                    job_post_text= f"job title: {title}. description: {clean_desc}",
                    model=model,
                    tokenizer=tokenizer
                )
                skill = extract_required_skills(
                    job_post_text= f"job title: {title}. description: {clean_desc}",
                    job_family=job_fam,
                    model=model,
                    tokenizer=tokenizer
                )
                extracted_info.append((id, title, clean_desc, job_fam, years, skill))
            else:
                extracted_info.append((id, title, clean_desc, None, None, None))
        except:
            pass # skip this role if encountered error

    job_posting_df_extracted = pd.DataFrame(
        data=extracted_info,
        columns=["id", "title", "desc", "job_fam", "years", "skill"]
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

