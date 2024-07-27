from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from src.gpt_sampler import ResumeMatcher
from src.sampler import clean_df

if __name__ == "__main__":


    resume_df = pd.read_csv(os.path.join("data", "resume_data_20240709115020.csv"))[["job_family", "resume", "years_of_experience", "skills_experience"]]
    # filter down to smaller subset
    # resume_df = resume_df[resume_df["years_of_experience"]==5]
    resume_df.dropna()
    

    resume_df = resume_df[resume_df["job_family"]=="software engineer"]
    resume_df = resume_df[resume_df["resume"].str.len() < 3000]

    job_post_df = pd.read_csv(os.path.join("data", "job_posting_transformed_20240708012832.csv"))
    job_post_df = job_post_df[job_post_df["years"]==5]
    job_post_df = job_post_df[job_post_df["job_fam"]=="software engineer"] # there is a typo
    job_post_df = job_post_df[job_post_df["desc"].str.len() < 3000]

    

    pairwise = job_post_df.iloc[0:1,:].join(
        resume_df
        , how="cross"
    )

    job_desc_col = pairwise["desc"]
    resume_col = pairwise["resume"]
    
    client = OpenAI()

    data_list = []

    for job_desc, resume in tqdm(zip(job_desc_col,resume_col), desc="Matching", ncols=100, total=pairwise.shape[0]):
        matcher = ResumeMatcher(client, job_desc, resume)
        output = matcher.check_fit()
        summ_requirement = matcher.required_skills
        summ_resume_skills = matcher.resume_skills
        data_list.append((
            job_desc, 
            resume, 
            summ_requirement, 
            summ_resume_skills,
            output["answer"], 
            output["justification"], 
            output["confidence"],
        ))
    
    output_df = pd.DataFrame(
        data=data_list, 
        columns=[
            "job_desc",
            "resume",
            "requirement",
            "resume_skills",
            "label",
            "justification",
            "confidence",
        ]
    )

    output_path = os.path.join(
        os.getcwd(), 
        "data", 
        f"gpt_match_output_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    output_df.to_csv(output_path, index=False)
    
