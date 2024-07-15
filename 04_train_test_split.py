import pandas as pd
from sklearn.model_selection import train_test_split
import os

match_df = pd.read_csv(os.path.join(os.getcwd(), "data", "match_output_20240712065719.csv"))

unique_jobs = match_df['job_description_text'].unique()

train_jobs, test_jobs = train_test_split(unique_jobs, test_size=0.2, random_state=123)

train_df = match_df[match_df['job_description_text'].isin(train_jobs)]
test_df = match_df[match_df['job_description_text'].isin(test_jobs)]

train_df.to_csv(os.path.join(os.getcwd(), "data", "train.csv"), index=False)
test_df.to_csv(os.path.join(os.getcwd(), "data", "test.csv"), index=False)
