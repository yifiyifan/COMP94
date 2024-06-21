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

# Set the path to the Edge binary (optional)
edge_options = Options()
# edge_options.binary_location = "C:/Path/To/Your/Edge/Application/msedge.exe"  # Uncomment and set the path if needed

# Use EdgeDriverManager to install the appropriate driver
service = Service(EdgeChromiumDriverManager().install())
driver = webdriver.Edge(service=service, options=edge_options)

job_list = ['HR', 'designer', 'Information-Technology',
       'Teacher', 'Advocate', 'Business-Development',
       'Healthcare', 'Fitness', 'Agriculture', 'BPO', 'Sales', 'Consultant',
       'Digital-Media', 'Automobile', 'Chef', 'Finance',
       'Apparel', 'Engineering', 'Accountant', 'Construction',
       'Public-Relations', 'Banking', 'Arts', 'Aviation']

df = pd.DataFrame()
category = []
link = []

for job in job_list:
    JOB = job.lower()
    for i in range(1,13):   # INCREASE THE RANGE TO GET MORE RESUME DATA
        PAGE = str(i)
        URL = "https://www.livecareer.com/resume-search/search?jt=" + JOB + "&bg=85&eg=100&comp=&mod=&pg=" + PAGE
        driver.get(URL)
        a_tags_in_div = driver.find_elements(By.CSS_SELECTOR, 'div a')

        for a in a_tags_in_div:
            if a.get_attribute('class') == "sc-1dzblrg-0 caJIKu sc-1os65za-2 jhoVRR":
                category.append(JOB)
                link.append(a.get_attribute('href'))

df["Category"] = category
df["link"] = link

import hashlib
def id(x):
    return int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)

df["id"] = df["link"].apply(id)

df["Resume"] = ""
df["Raw_html"] = ""

for i in range(df.shape[0]):
    url = df.link[i]
    driver.get(url)
#     time.sleep(0.5)                  #ADDED DELAY, CAN BE REMOVED
    x = driver.page_source
    x = x.replace(">","> ")
    soup = bs4.BeautifulSoup(x, 'html.parser')
    div = soup.find("div", {"id": "document"})
    df.Raw_html[i] = div
    try:
        df.Resume[i] = div.text
    except:
#         ADD EXCEPTION IF REQUIRED
        pass

df.to_csv("result.csv", index=False)