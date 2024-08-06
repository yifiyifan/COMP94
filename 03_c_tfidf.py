import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":


    resume_df = pd.read_csv(os.path.join("data", "example_input", "resume-40-data-creation-examples.csv"))[["job_family", "resume", "years_of_experience", "skills_experience"]]
    # resume_df = pd.read_csv(os.path.join("data", "example_input", "resume-10-holdout-examples.csv"))[["job_family", "resume", "years_of_experience", "skills_experience"]]
    
    resume_df = resume_df.dropna()
    
    job_post_df = pd.read_csv(os.path.join("data", "example_input", "jd-15-data-creation-examples.csv"))
    # job_post_df = pd.read_csv(os.path.join("data", "example_input", "jd-5-holdout-examples.csv"))


    pairwise = job_post_df.join(
        resume_df
        , how="cross"
    )

    job_desc_col = pairwise["desc"]
    resume_col = pairwise["resume"]
    


    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Combine the two columns into a single list for vectorization
    combined_text = job_desc_col.tolist() + resume_col.tolist()

    # Fit and transform the combined text data into TF-IDF scores
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    # Split the TF-IDF matrix back into the original two sets
    tfidf_job_desc = tfidf_matrix[:pairwise.shape[0]]
    tfidf_resume = tfidf_matrix[pairwise.shape[0]:]

    # Compute the cosine similarity between pairs of values
    cosine_similarities = []
    for i in tqdm(range(pairwise.shape[0])):
        cosine_sim = cosine_similarity(tfidf_job_desc[i], tfidf_resume[i])
        cosine_similarities.append(cosine_sim[0][0])

    # # Add the cosine similarities to the DataFrame
    pairwise['cosine_similarity'] = cosine_similarities

    output_path = os.path.join(
        os.getcwd(), 
        "data", 
        f"tfidf_match_output_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    pairwise.to_csv(output_path, index=False)
