# from src.scrap_data import (
#     scrape_resume_link,
#     get_resume_info,
# )

# modules in project
import logging.config

from src.scrape_resume_data import (
    set_up_driver,
    scrape_resume_link,
    get_resume_info
)
from src.utils import (
    format_execution_time,
    initialise_job_exp_grid
)
from src.configs import (
    CONFIG,
    LOGGING_CONFIG,
)

# import required libraries 
import os 
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import logging
import time
import pandas as pd


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def scrape_resume():
    """
        for each job family
    """
    job_fam_list = CONFIG['resume_scraping_params']['job_family']
    exp_list = [(val['start'], val['end']) for _, val in CONFIG['resume_scraping_params']['experience_band'].items()]
    grid = initialise_job_exp_grid(job_fam_list, exp_list)

    links = []
    for i in tqdm(range(grid.shape[0]), "Scraping", ncols=100, total=grid.shape[0]):
        job_fam = str(grid['job_family'][i])
        exp_lower = str(grid['exp_lower'][i])
        exp_upper = str(grid['exp_upper'][i])
        logger.debug(f"scraping for {job_fam} between {exp_lower} and {exp_upper} years of experience")
        tmp_list = scrape_resume_link(
            job_family=job_fam,
            exp_lower=exp_lower,
            exp_upper=exp_upper,
            max_page=CONFIG['resume_scraping_params']['pages_per_search']
        )
        links.append("|".join(tmp_list))
    
    links_df = grid.copy()
    links_df["links"] = links
    output_path = os.path.join(
        os.getcwd(), 
        "data", 
        f"resume_links_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    links_df.to_csv(output_path, index=False)
    return links_df

def resume_link_to_data(links_df):

    scraped_tuples = []
    for i in tqdm(range(links_df.shape[0]), "Processing group", ncols=100, total=links_df.shape[0]):
        
        job_fam = str(links_df['job_family'][i])
        url_list = str(links_df['links'][i]).split("|")
        for url in tqdm(url_list, "Scraping resume details", ncols=100, total=len(url_list)):
            driver = set_up_driver()
            job_title, resume, years, skill_exp = get_resume_info(driver=driver, link=url)
            driver.quit()
            scraped_tuples.append((job_fam, job_title, resume, years, skill_exp))
    
    output_df = pd.DataFrame(data=scraped_tuples, columns=["job_family", "job_title", "resume", "years_of_experience", "skills_experience"])
    output_path = os.path.join(
        os.getcwd(), 
        "data", 
        f"resume_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    output_df.to_csv(output_path, index=False)

    return output_df

if __name__ == "__main__":

    start_time = time.time()
    df = scrape_resume()
    resume_df = resume_link_to_data(df)
    end_time = time.time()

    logger.info("Processed complete in %s" % (format_execution_time(start_time, end_time)))
