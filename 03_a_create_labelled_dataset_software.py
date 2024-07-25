from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from src.gpt_sampler import ResumeMatcher

if __name__ == "__main__":
    dev_exmaples = pd.read_csv("data/golden_example.csv")
    print(dev_exmaples.head())
    
    client = OpenAI()

    data_list = []

    for job_desc, resume in tqdm(zip(dev_exmaples["job_desc"], dev_exmaples["resume"]), desc="Matching", ncols=100, total=dev_exmaples.shape[0]):
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
            float(output["confidence"]),
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
            "confidence"
        ]
    )

    output_path = os.path.join(
        os.getcwd(), 
        "data", 
        f"match_output_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    output_df.to_csv(output_path, index=False)
    
