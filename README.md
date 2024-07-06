# COMP9491

## Methodology to create the dataset 

### Collecting job description data 

1. Download the LinkedIn Job Post 2023-2024 dataset 
2. Filter down to a smaller subset of job post that contain a list of key words 
    1. This is currently done in excel, TODO: explore fuzzy match logic and take subset with python 
3. Use flan-t5 to classify job family 
4. For jobs classified into job family of interest, extract other information including 
    1. Years of experience required 
    2. Level (junior / senior)
    3. Key skills required 

### Collecting resume data

1. From a list of key words + level of experience combinatio, generate url that will show a list of resume examples 
2. Search in the page html to identify urls to resume examples shown on the page 
3. Loop through each resume url to extract 
    1. Title 
    2. Resume text 
    3. Years of experience
    4. Skills 

### Hueristic matching

Attributes of good fit 
- High sentence encoder similarity between title and job family x level 
- High similairity between skill of applicant and required skills 
- Years of experience > required years of experience 

say we have 300 jobs in total 
