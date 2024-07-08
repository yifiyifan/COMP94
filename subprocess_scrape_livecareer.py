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
from src.scrape_resume_data import (
    scrape_resume_link,
    get_resume_info
)
from src.utils import (
    clean_string,
    format_execution_time,
    initialise_job_exp_grid
)
from src.configs import (
    CONFIG,
    LOGGING_CONFIG,
    JOB_POSTING_PATH,
    FLAN_T5_MODEL_NAME, # probably not needed ?
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

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By

from webdriver_manager.microsoft import EdgeChromiumDriverManager
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import bs4
import requests
import re
import hashlib

pattern_job_title = r'.+(?=\sresume example with)'
pattern_num_year = r'(?<=resume example with\s)[0-9]+'
pattern = re.compile(r'^SECTION_(SUMM|HILT|EDUC|EXPR|SKLL).+$')


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def scrape_resume():
    """
        for each job family
    """
    job_fam_list = CONFIG['resume_scraping_params']['job_family']
    exp_list = [(val['start'], val['end']) for _, val in CONFIG['resume_scraping_params']['experience_band'].items()]
    grid = initialise_job_exp_grid(job_fam_list, exp_list)



if __name__ == "__main__":

    start_time = time.time()

    logger.info("Starting the process")
    
    scrape_resume()
    # # Step 1: extract information from job posts 
    # logger.info("Extracting info from job posts...")
    # job_post_df_extracted = extract_job_post_info(
    #     input_path=JOB_POSTING_PATH,
    #     model_name=FLAN_T5_MODEL_NAME,
    # )
    # logger.info("Extraction from job post finished")

    # output_path = os.path.join(
    #     os.getcwd(), 
    #     "data", 
    #     f"job_posting_transformed_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    # )
    
    # logger.info("Saving job post extraction output to %s" % (output_path))
    # job_post_df_extracted.to_csv(output_path,index=False)

    end_time = time.time()

    logger.info("Processed complete in %s" % (format_execution_time(start_time, end_time)))
