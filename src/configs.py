import yaml
import os 

CONFIG = yaml.load(
    open(os.path.join(os.getcwd(), "config.yaml"), "r+"),
    Loader = yaml.FullLoader
)

LOGGING_CONFIG = yaml.load(
    open(os.path.join(os.getcwd(), "logging_config.yaml"), "r+"),
    Loader = yaml.FullLoader
)

JOB_POSTING_PATH = os.path.join(os.getcwd(), "data", "job_postings_filtered_to_keywords.csv")
FLAN_T5_MODEL_NAME = "google/flan-t5-large"