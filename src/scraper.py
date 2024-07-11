from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from fake_useragent import UserAgent

from webdriver_manager.microsoft import EdgeChromiumDriverManager
from bs4 import BeautifulSoup
import re
import hashlib
import time

pattern_job_title = r'.+(?=\sresume example with)'
pattern_num_year = r'(?<=resume example with\s)[0-9]+'
pattern = re.compile(r'^SECTION_(SUMM|HILT|EDUC|EXPR|SKLL).+$')
pattern_skill_exp = re.compile(r'^SECTION_(EXPR|SKLL).+$')

def id(x):
    return int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)

def set_up_driver(headless:bool=False) -> webdriver.Edge:
    edge_options = Options()
    if headless:
        edge_options.add_argument("--headless")  
        edge_options.add_argument("--disable-gpu")  
        edge_options.add_argument("--window-size=1920x1080") 
    ua = UserAgent()
    userAgent = ua.random
    edge_options.add_argument(f'user-agent={userAgent}')
    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=edge_options)
    return driver 

def scrape_resume_link(
    job_family:str, 
    exp_lower:str, 
    exp_upper:str,
    min_rating:int=85, 
    max_rating:int=100, 
    max_page:int=5
):
    """Return a list of resume links matching the search criteria"""
    result = []
    search_keyword = job_family.lower().replace(" ","%20")
    for i in range(max_page): 
        page_no = str(i+1)
        url = f"https://www.livecareer.com/resume-search/search?jt={search_keyword}&be={exp_lower}&ee={exp_upper}&bg={str(min_rating)}&eg={str(max_rating)}&comp=&mod=&pg={page_no}"
        driver=set_up_driver()
        driver.get(url)
        time.sleep(0.1)
        a_tags_in_div = driver.find_elements(By.CSS_SELECTOR, 'div a') 
        for a in a_tags_in_div:
            if a.get_attribute('class') == "sc-1dzblrg-0 caJIKu sc-1os65za-2 jhoVRR":
                result.append(a.get_attribute('href'))
        driver.quit()
    return result

def get_resume_info(driver:webdriver.Edge, link:str):
    """Return job title, resume text, years of experience"""
    try:
        # tmp_resume = '' 
        # tmp_years = '' 
        # tmp_job_title = ''
        driver.get(link)
        time.sleep(0.1)
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        title_tag = soup.find('title')
        if title_tag and '404 - Page Not Found' in title_tag.text:
            return None, None, None, None
        divs = soup.find_all('div', id=pattern)
        tmp_resume = "\n".join([div.get_text(strip=True)for div in divs])
        divs_skill_exp = soup.find_all('div', id=pattern_skill_exp)
        tmp_skill_exp = "\n".join([div.get_text(strip=True)for div in divs_skill_exp])
        page_title = soup.find_all('h2', class_='title')[0].get_text(strip=False)
        tmp_job_title = re.findall(pattern_job_title, page_title)[0]
        tmp_years = re.findall(pattern_num_year, page_title)[0]
        return tmp_job_title, tmp_resume, tmp_years, tmp_skill_exp
    except Exception as e:
        return None, None, None, None