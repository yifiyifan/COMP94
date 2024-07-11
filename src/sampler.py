import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from dataclasses import dataclass
import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

from src.configs import (
    CONFIG,
    LOGGING_CONFIG,
    FLAN_T5_MODEL_NAME,
)

from src.extractor import (
    extract_stated_skills
)

def clean_df(input_df:pd.DataFrame, job_fam_col:str, target_col:str, similarity_threshold:float=0.99):
    # remove failed scrape 
    output_df = input_df.dropna(how="any")

    # remove deduplicate 
    output_df = output_df.drop_duplicates()
    
    # remove over similar examples 
    kept_dfs = []
    for job_fam in CONFIG["job_family"]:
        subset:pd.DataFrame = output_df[output_df[job_fam_col]==job_fam]
        # subset["orig_index"]=subset.index
        subset.reset_index(drop=True)
        documents = subset[target_col]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        id_to_keep = []
        duplicate_flags = np.zeros(len(documents))

        for i in range(len(documents)):
            if duplicate_flags[i] == 0:
                # Mark this document as a keeper
                id_to_keep.append(i)
                # Compare this document with all subsequent documents
                for j in range(i + 1, len(documents)):
                    if cosine_sim_matrix[i][j] >= similarity_threshold:
                        # Mark this document as a duplicate
                        duplicate_flags[j] = 1

        retained_rows = subset.iloc[id_to_keep]
        kept_dfs.append(retained_rows)
    output_df = pd.concat(kept_dfs, axis=0).reset_index(drop=True)
    return output_df

def tfidf_sim(x, y):
    documents = [str(x),str(y)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim_score = cosine_similarity(tfidf_matrix, tfidf_matrix)[0,1]
    return cosine_sim_score

@dataclass
class ResumeSampler():

    # job_title, job_desc, min_years, requirement, max_year
    job_post:pd.DataFrame
    # job_family, resume, years_of_experience, skills_experience
    resume_df:pd.DataFrame

    def cross_join(self):
        output = self.job_post.join(
            other= self.resume_df,
            how="cross"
        )
        return output
    
    def mistmached_job_fam(self, sample_n=2000):
        df = self.cross_join()
        df_all_mismatch = df[df["advertised_job_fam"]!=df["job_family"]]
        if df_all_mismatch.shape[0] > sample_n:
            return df_all_mismatch.sample(n=sample_n)
        else:
            return df_all_mismatch
    
    def mistmatched_years_of_experience(self, sample_n=2000):
        df = self.cross_join()
        df = df[df["advertised_job_fam"]==df["job_family"]]
        underqualify = list(df["years_of_experience"]<df["min_years"])
        overqualify = list(df["years_of_experience"]>df["max_years"])
        index_not_meet_criteria = (underqualify) or (overqualify)
        df_all_mismatch = df[index_not_meet_criteria]
        if df_all_mismatch.shape[0] > sample_n:
            return df_all_mismatch.sample(n=sample_n)
        else:
            return df_all_mismatch
    
    def good_and_potential_match(self, sample_n_pot=2000, sample_n_good=2000):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = T5Tokenizer.from_pretrained(FLAN_T5_MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(FLAN_T5_MODEL_NAME).to(device)

        df = self.cross_join()
        return_cols = df.columns.to_list()
        df = df[df["advertised_job_fam"]==df["job_family"]]
        underqualify = list(df["years_of_experience"]<df["min_years"])
        overqualify = list(df["years_of_experience"]>df["max_years"])
        meet_criteria = [not (u or o) for u, o in zip(underqualify, overqualify)]
        df_all_match = df[meet_criteria]

        subset_list = []

        for job_fam in tqdm(CONFIG["job_family"], desc="Sampling by job family", ncols=100, total=len(CONFIG["job_family"])):
            subset:pd.DataFrame = df_all_match[df_all_match["job_family"]==job_fam].reset_index(drop=True)
            # tmp_sum_col = []
            tmp_sim_col = []
            for i in tqdm(range(subset.shape[0]), desc="processing", ncols=100, total=subset.shape[0]):
                try: 
                    sim_score = extract_stated_skills(
                        str(subset["requirement"][i]),
                        str(subset["skills_experience"][i]),
                        model,
                        tokenizer
                    )
                    # sim_score = tfidf_sim(summ_skill, subset["requirement"][i])
                    # tmp_sum_col.append(summ_skill)
                    tmp_sim_col.append(sim_score)
                except Exception as e:
                    print(e)
                    # tmp_sum_col.append(None)
                    tmp_sim_col.append(None)
            # subset["summ_skills"] = tmp_sum_col
            subset["similarity"] = tmp_sim_col
            subset_list.append(subset.copy())

        output_df = pd.concat(subset_list, axis=0).reset_index(drop=True)

        output_df.to_csv(os.path.join("data", "resume_similarity.csv"), index=False)

        good_match_mask = output_df["similarity"].apply(lambda x: x =="4 - met all requirements")
        requirement_len_mask = output_df["requirement"].apply(func=lambda x: len(x) >= 200) # some short descriptions are just not informative
        overall_mask = good_match_mask & requirement_len_mask

        good_match_df = output_df[overall_mask]
        potential_match_df = output_df[~overall_mask]

        if good_match_df.shape[0] > sample_n_good:
            good_match_df = good_match_df.sample(n=sample_n_good)
        if potential_match_df.shape[0] > sample_n_pot:
            potential_match_df = potential_match_df.sample(n=sample_n_pot)
        
        return good_match_df[return_cols], potential_match_df[return_cols]