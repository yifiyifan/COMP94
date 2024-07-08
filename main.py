# from src.scrap_data import (
#     scrape_resume_link,
#     get_resume_info,
# )

# modules in project
import logging.config
from src.sampler import (
    extract_job_title,
    extract_required_skills,
    extract_years_of_experience,
)
from src.utils import (
    clean_string,
    format_execution_time,
)
from src.configs import (
    CONFIG,
    LOGGING_CONFIG,
    JOB_POSTING_PATH,
    FLAN_T5_MODEL_NAME,
)

# import required libraries 
import os 
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
from tqdm.contrib.logging import tqdm_logging_redirect

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def extract_job_post_info(
    input_path:str,
    model_name:str,
):
    # set up 
    job_posting_df = pd.read_csv(input_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    job_id_col = job_posting_df["job_id"]
    job_postings_desc_col = job_posting_df["description"]
    job_title_col = job_posting_df["title"]
    extracted_info = []

    cnt_total_rows = job_posting_df.shape[0]

    with tqdm_logging_redirect():
        for id, title, desc in tqdm(zip(job_id_col, job_title_col, job_postings_desc_col), desc="Extracting",ncols=100, total=cnt_total_rows):
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
            except KeyboardInterrupt as e:
                raise e
            except:
                pass # skip this job post if encountered error
    
    
    cnt_extracted_rows = len(extracted_info)
    stat_message = "Count of job post: %d; extracted: %d" % (cnt_total_rows, cnt_extracted_rows)
    logger.info(stat_message)

    # return convert extracted info as dataframe 
    return pd.DataFrame(
        data=extracted_info,
        columns=["id", "title", "desc", "job_fam", "years", "skill"]
    )


if __name__ == "__main__":

    start_time = time.time()

    logger.info("Starting the process")
    
    # Step 1: extract information from job posts 
    logger.info("Extracting info from job posts...")
    job_post_df_extracted = extract_job_post_info(
        input_path=JOB_POSTING_PATH,
        model_name=FLAN_T5_MODEL_NAME,
    )
    logger.info("Extraction from job post finished")

    output_path = os.path.join(
        os.getcwd(), 
        "data", 
        f"job_posting_transformed_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    
    logger.info("Saving job post extraction output to %s" % (output_path))
    job_post_df_extracted.to_csv(output_path,index=False)

    end_time = time.time()

    logger.info("Processed complete in %s" % (format_execution_time(start_time, end_time)))
