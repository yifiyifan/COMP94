from openai import OpenAI
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

SYSTEM_MESSAGE_BRIEFING = """You are an experienced recruiter and you are screen resumes you received for a job openning. 
First you want to extract the key information from the resume and make them into the format below

[SUMMARY]
Headline summary of the candidate, including current job title, number of years of experience, if there are any leadership experience and any specialty

[EDUCATION]
This section is an one to two sentence summary of the education background and qualification of the candidate

[WORK EXPERIENCE]
This section is a list of the work experience shown in the resume in the format
- company name, job title (start date - end date): description of highlights, skills and capability demonstrated
- ...

[SKILLS]
THis seciton is a list of skills demonstrated by this resume. Also label the proficiency of each skill mentioned. 

"""

# SYSTEM_MESSAGE_SKILLS = (
#     "You are a recruiter who is assessing the goodness of fit of job applicants"
#     "by comparing the candidate's resume with the required and / or desirable skills and experience job description. "
#     "The skills to consider are:"
#     "\n\nProgramming Languages\n- General-Purpose: Python, Java, C++, C#, Go, Rust\n- Scripting: JavaScript, TypeScript, Bash, PowerShell"
#     "\n\nSoftware Development Practices\n- Version Control: Git\n- Agile Methodologies: Scrum, Kanban\n- DevOps: CI/CD, IaC\n- Testing: Unit Testing, Integration Testing, TDD\n- Documentation: API Documentation, Technical Writing"
#     "\n\nDevelopment Frameworks and Libraries\n- Web Development: React, Angular, Vue.js, Django, Flask\n- Mobile Development: Swift, Kotlin, React Native\n- Machine Learning: TensorFlow, PyTorch, Scikit-Learn\n- Data Analysis: Pandas, NumPy"
#     "\n\nSoftware Design and Architecture\n- Design Patterns: Singleton, Factory, Observer\n- Architectural Styles: Microservices, Monolithic, SOA\n- UML Diagrams: Class Diagrams, Sequence Diagrams"
#     "\n\nDatabases\n- Relational: MySQL, PostgreSQL, SQL Server\n- NoSQL: MongoDB, Redis\n- Design: Normalization, Indexing"
#     "\n\nCloud Computing\n- Platforms: AWS, Azure, GCP\n- Containerization: Docker, Kubernetes\n- Serverless: AWS Lambda, Azure Functions"
#     "\n\nSecurity\n- Application Security: OWASP Top 10, Secure Coding\n- Network Security: Firewalls, VPNs\n- Cryptography: Encryption Algorithms, Hashing"
#     "\n\nNetworking\n- Protocols: HTTP/HTTPS, TCP/IP\n- Tools: Wireshark, Postman\n- Concepts: Load Balancing, Caching"
#     "\n\nOperating Systems and Environments\n- OS: Windows, Linux, macOS"
#     "\n\nSoft Skills\n- Communication: Verbal, Written\n- Collaboration: Teamwork, Mentorship\n- Project Management: Time Management, Problem-Solving"
#     "\n\nEmerging Technologies\n- Blockchain: Smart Contracts, Ethereum\n- AI/ML: Neural Networks, NLP, Computer Vision"
#     "\n\nTools and IDEs\n- IDEs: VS Code, IntelliJ IDEA, Eclipse\n- Build Tools: Maven, Gradle\n- CI Tools: Jenkins"
# )



class ResumeCleaner():

    def __init__(self, open_ai_client:OpenAI, resume:str) -> None:
        self.client = open_ai_client
        self.model = "gpt-4o-mini"
        self._messages = []   
        self.resume = resume
    
    # utilities 
    def _query_model(self):
        """Query model with current parameters and messages"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            # parameterise later
            temperature=0.8,
            max_tokens=2048, # this needs to be sufficiently large
            top_p=1,
            frequency_penalty=0.25, # penalise repeated words
            presence_penalty=0.6    # penalise words already mentioned, idea is to give distinct skills
        )
        return response.choices[0].message.content
    
    def add_system_msg(self, text:str):
        msg = {
                "role": "system",
                "content": [{"type": "text", "text": text}]
            }
        self._messages.append(msg)

    def add_user_msg(self, template:str, parameterisation:dict=None):
        if parameterisation is not None:
            user_msg = template.format(**parameterisation)
        else:
            user_msg = template
        msg = {
                "role": "user",
                "content": [{"type": "text", "text": user_msg}]
            }
        self._messages.append(msg)


    def add_assistant_msg(self, text:str):
        msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": text}]
            }
        self._messages.append(msg)

def format_resume(raw_resume:str):

    client = OpenAI()
    cleaner = ResumeCleaner(client,raw_resume)
    cleaner.add_system_msg(SYSTEM_MESSAGE_BRIEFING)
    cleaner.add_user_msg(raw_resume)
    return cleaner._query_model()

if __name__ == "__main__":
    # schema: job_desc	resume	requirement	resume_skills	label	justification
    # base_df = pd.read_csv(os.path.join("data", "gpt_match_output_20240726182238.csv"))
    base_df = pd.read_csv(os.path.join("data", "final_gpt_match_output_20240727204313.csv"))
    
    base_df = base_df.dropna()

    formatted_resume = []
    for res in tqdm(base_df["resume"], desc="Formatting", ncols=100, total=base_df.shape[0]):
        try:
            formatted_resume.append(format_resume(res))
        except Exception:
            formatted_resume.append(None)

    base_df["formatted_resume"] = formatted_resume

    output_path = os.path.join(
        os.getcwd(), 
        "data", 
        f"final_gpt_match_output_clean_res_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    base_df.to_csv(output_path, index=False)
    





# SYSTEM_MESSAGE_SKILLS = """This is the skills and capability framework of a data analyst type role
# Technical Skills
#  - Programming Languages: SQL, Python, R, SAS, MATLAB
#  - Data Visualization: Tableau, Power BI, Looker, D3.js, Matplotlib/Seaborn (Python libraries)
#  - Data Manipulation and Analysis: Pandas (Python library), Excel, NumPy (Python library), SQL queries, Data cleaning and transformation
#  - Database Management: SQL Server, MySQL, PostgreSQL, Oracle, NoSQL databases (e.g., MongoDB)
#  - Statistical Analysis: Hypothesis testing, Regression analysis, A/B testing, Time series analysis

# Analytical Skills
#  - Data Interpretation: Identifying trends and patterns, Drawing insights from data, Making data-driven recommendations
#  - Problem Solving: Analyzing complex problems, Developing solutions based on data
#  - Critical Thinking: Evaluating data sources, Assessing data quality, Questioning assumptions

# Business and Domain Knowledge
#  - Industry Knowledge: Understanding specific industry metrics and KPIs, Awareness of industry-specific challenges and opportunities
#  - Business Acumen: Aligning data analysis with business goals, Communicating insights to stakeholders in a business context

# Soft Skills
#  - Communication: Presenting data findings to non-technical audiences, Writing clear and concise reports
#  - Collaboration: Working effectively in teams, Coordinating with other departments
#  - Project Management: Managing data projects from start to finish, Time management and prioritization

# Tools and Software
#  - Data Analytics Platforms: Google Analytics, Adobe Analytics
#  - ETL Tools: Alteryx, Apache NiFi, Talend
#  - Big Data Technologies: Hadoop, Spark, Hive

# Data Engineering Skills (Optional but Beneficial)
#  - Data Warehousing: Designing and managing data warehouses, ETL pipeline development
#  - Cloud Computing: AWS (e.g., Redshift, S3), Google Cloud Platform (e.g., BigQuery), Microsoft Azure (e.g., Azure SQL Database)

# Machine Learning and Advanced Analytics (Optional but Beneficial)
#  - Machine Learning Algorithms: Linear Regression, Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Naive Bayes, Clustering (e.g., K-Means, Hierarchical Clustering), Principal Component Analysis (PCA), Ensemble Methods (e.g., Gradient Boosting, AdaBoost), Neural Networks (e.g., MLP, CNN, RNN), Natural Language Processing (e.g., Text Classification, Sentiment Analysis)
#  - Predictive Analytics: Building predictive models, Forecasting techniques
# """