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

def id(x):
    return int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)

# Set the path to the Edge binary (optional)
edge_options = Options()
# edge_options.binary_location = "C:/Path/To/Your/Edge/Application/msedge.exe"  # Uncomment and set the path if needed

# Use EdgeDriverManager to install the appropriate driver
service = Service(EdgeChromiumDriverManager().install())
driver = webdriver.Edge(service=service, options=edge_options)

job_list = [
    'accountant', 
    'chef', 
    'data analyst', 
    'software engineer', 
    'vet',
    'salesperson',
]
MAX_PAGE = 2

df = pd.DataFrame()
category = []
link = []
job_title=[]
years_of_experience = []
resume_text = []

for job in job_list:
    JOB = job.lower().replace(" ","%20")
    for i in range(MAX_PAGE):   # INCREASE THE RANGE TO GET MORE RESUME DATA
        PAGE = str(i+1)
        URL = "https://www.livecareer.com/resume-search/search?jt=" + JOB + "&bg=85&eg=100&comp=&mod=&pg=" + PAGE
        driver.get(URL)
        a_tags_in_div = driver.find_elements(By.CSS_SELECTOR, 'div a')

        for a in a_tags_in_div:
            if a.get_attribute('class') == "sc-1dzblrg-0 caJIKu sc-1os65za-2 jhoVRR":
                category.append(job)
                link.append(a.get_attribute('href'))

df["Category"] = category
df["link"] = link
df["id"] = df["link"].apply(id)

def scrape_resume_link(driver:webdriver.Edge, job_family:str, exp_level:str, min_rating:int=85, max_rating:int=100, max_page:int=5):
    """Return a list of resume links matching the search criteria"""
    result = []
    search_keyword = " ".join([exp_level, job_family]).lower().replace(" ","%20")
    for i in range(max_page):   
        page_no = str(i+1)
        url = f"https://www.livecareer.com/resume-search/search?jt={search_keyword}&bg={str(min_rating)}&eg={str(max_rating)}&comp=&mod=&pg={page_no}"
        driver.get(url)
        a_tags_in_div = driver.find_elements(By.CSS_SELECTOR, 'div a') 
        for a in a_tags_in_div:
            if a.get_attribute('class') == "sc-1dzblrg-0 caJIKu sc-1os65za-2 jhoVRR":
                result.append(a.get_attribute('href'))
    return result

def get_resume_info(driver:webdriver.Edge, link:str):
    """Return job title, resume text, years of experience"""
    try:
        # tmp_resume = '' 
        # tmp_years = '' 
        # tmp_job_title = ''
        driver.get(link)
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        divs = soup.find_all('div', id=pattern)
        tmp_resume = "\n".join([div.get_text(strip=True)for div in divs])
        page_title = soup.find_all('h2', class_='title')[0].get_text(strip=False)
        tmp_job_title = re.findall(pattern_job_title, page_title)[0]
        tmp_years = re.findall(pattern_num_year, page_title)[0]
    except Exception:
        return None, None, None
    finally:
        return tmp_job_title, tmp_resume, tmp_years

for l in df['link']:
    try:        
        title, resume, num_years = get_resume_info(driver, l)
    except Exception:
        pass
    finally:
        job_title.append(title)
        resume_text.append(resume)
        years_of_experience.append(num_years)

df['job_title'] = job_title
df['resume_text'] = resume_text
df['years_of_experience'] = years_of_experience

print(f"scraped {df.shape[0]} resume")

df.to_csv("result.csv", index=False)


