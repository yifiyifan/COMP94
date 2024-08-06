from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from src.gpt_sampler import ResumeMatcher
from src.sampler import clean_df

if __name__ == "__main__":


    # resume_df = pd.read_csv(os.path.join("data", "example_input", "resume-40-data-creation-examples.csv"))[["job_family", "resume", "years_of_experience", "skills_experience"]]
    resume_df = pd.read_csv(os.path.join("data", "example_input", "resume-10-holdout-examples.csv"))[["job_family", "resume", "years_of_experience", "skills_experience"]]
    
    resume_df = resume_df.dropna()
    
    # job_post_df = pd.read_csv(os.path.join("data", "example_input", "jd-15-data-creation-examples.csv"))
    job_post_df = pd.read_csv(os.path.join("data", "example_input", "jd-5-holdout-examples.csv"))


    pairwise = job_post_df.join(
        resume_df
        , how="cross"
    )

    job_desc_col = pairwise["desc"]
    resume_col = pairwise["resume"]
    
    client = OpenAI()

    data_list = []
    cnt = 0 
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
        cnt += 1 
        if cnt % 40 == 0:
            tmp_df = output_df = pd.DataFrame(
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
            tmp_output_path = os.path.join(
                os.getcwd(), 
                "data", 
                f"holdout_gpt_match_output_checkpoint_{cnt}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            )
            tmp_df.to_csv(tmp_output_path, index=False)
    
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
        f"holdout_gpt_match_output_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    output_df.to_csv(output_path, index=False)
    
