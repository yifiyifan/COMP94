import logging.config
import pandas as pd 
import os 
import logging
from datetime import datetime

from src.sampler import (
    clean_df,
    ResumeSampler,
)

from src.configs import (
    CONFIG,
    LOGGING_CONFIG,
)

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    resume_df = pd.read_csv(os.path.join("data", "resume_data_20240709115020.csv"))[["job_family", "resume", "years_of_experience", "skills_experience"]]
    # resume_df = pd.read_csv(os.path.join("data", "resume_data_dev2.csv"))[["job_family", "resume", "years_of_experience", "skills_experience"]]

    cleaned_resume_df = clean_df(resume_df, "job_family", "skills_experience")

    print(resume_df.shape[0])
    print(cleaned_resume_df.shape[0])

    job_post_df = pd.read_csv(os.path.join("data", "job_posting_transformed_20240708012832.csv"))
    # job_post_df = pd.read_csv(os.path.join("data", "job_posting_transformed_dev.csv"))
    cleaned_job_post_df = clean_df(job_post_df, "job_fam", "desc")

    cleaned_job_post_df = cleaned_job_post_df[[
        "title","desc","job_fam","years","skill"
    ]].assign(
        max_years= cleaned_job_post_df["years"]*1.5 # assumed rule for overqualification
    ).rename(
        columns={
            "title":"job_title",
            "desc":"job_desc",
            "job_fam":"advertised_job_fam",
            "years":"min_years",
            "skill":"requirement"
        }
    )

    print(job_post_df.shape[0])
    print(cleaned_job_post_df.shape[0])

    print(cleaned_job_post_df.head())
    print(cleaned_resume_df.head())

    sampler = ResumeSampler(
        cleaned_job_post_df,
        cleaned_resume_df
    )

    mismatch_job_fam = sampler.mistmached_job_fam()
    mismatch_years = sampler.mistmatched_years_of_experience()
    good_match, potential_match = sampler.good_and_potential_match()
    
    mismatch_job_fam["label"]="No Fit"
    mismatch_years["label"]="No Fit"
    potential_match["label"]="Potential Fit"
    good_match["label"]="Good Fit"

    output_col = ["resume", "job_desc", "label"]

    output_df_detailed = pd.concat(
        [mismatch_job_fam, mismatch_years, potential_match, good_match],
        axis=0
    )

    output_path_detailed = os.path.join(
        os.getcwd(), 
        "data", 
        f"match_output_detailed_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    output_df_detailed.to_csv(output_path_detailed, index=False)
    
    output_df = output_df_detailed.loc[
        :,
        output_col
    ].rename(
        columns={
            "resume":"resume_text",
            "job_desc":"job_description_text"
        }
    )

    output_path = os.path.join(
        os.getcwd(), 
        "data", 
        f"match_output_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    output_df.to_csv(output_path, index=False)