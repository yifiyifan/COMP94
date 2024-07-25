import os 
from openai import OpenAI
import pandas as pd
import json
from typing import Dict

from dotenv import load_dotenv

from pprint import pprint

load_dotenv()

SYSTEM_MESSAGE = (
    "You are a recruiter who is assessing the goodness of fit of job applicants"
    "by comparing the candidate's resume with the required and / or desirable skills and experience job description. "
    "The skills to consider are:"
    "\n\nProgramming Languages\n- General-Purpose: Python, Java, C++, C#, Go, Rust\n- Scripting: JavaScript, TypeScript, Bash, PowerShell"
    "\n\nSoftware Development Practices\n- Version Control: Git\n- Agile Methodologies: Scrum, Kanban\n- DevOps: CI/CD, IaC\n- Testing: Unit Testing, Integration Testing, TDD\n- Documentation: API Documentation, Technical Writing"
    "\n\nDevelopment Frameworks and Libraries\n- Web Development: React, Angular, Vue.js, Django, Flask\n- Mobile Development: Swift, Kotlin, React Native\n- Machine Learning: TensorFlow, PyTorch, Scikit-Learn\n- Data Analysis: Pandas, NumPy"
    "\n\nSoftware Design and Architecture\n- Design Patterns: Singleton, Factory, Observer\n- Architectural Styles: Microservices, Monolithic, SOA\n- UML Diagrams: Class Diagrams, Sequence Diagrams"
    "\n\nDatabases\n- Relational: MySQL, PostgreSQL, SQL Server\n- NoSQL: MongoDB, Redis\n- Design: Normalization, Indexing"
    "\n\nCloud Computing\n- Platforms: AWS, Azure, GCP\n- Containerization: Docker, Kubernetes\n- Serverless: AWS Lambda, Azure Functions"
    "\n\nSecurity\n- Application Security: OWASP Top 10, Secure Coding\n- Network Security: Firewalls, VPNs\n- Cryptography: Encryption Algorithms, Hashing"
    "\n\nNetworking\n- Protocols: HTTP/HTTPS, TCP/IP\n- Tools: Wireshark, Postman\n- Concepts: Load Balancing, Caching"
    "\n\nOperating Systems and Environments\n- OS: Windows, Linux, macOS"
    "\n\nSoft Skills\n- Communication: Verbal, Written\n- Collaboration: Teamwork, Mentorship\n- Project Management: Time Management, Problem-Solving"
    "\n\nEmerging Technologies\n- Blockchain: Smart Contracts, Ethereum\n- AI/ML: Neural Networks, NLP, Computer Vision"
    "\n\nTools and IDEs\n- IDEs: VS Code, IntelliJ IDEA, Eclipse\n- Build Tools: Maven, Gradle\n- CI Tools: Jenkins"
)

JOB_DESC_USER_INPUT_TEMPLATE = (
    "[job description start]\n{placeholder}\n[job description end]"
    "\n\nWhat skills are required and / or desired to be successful candidate for this position?  "
    "Return output as a paragraph"
)

RESUME_SKILLS_USER_INPUT_TEMPLATE = (
    "[resume start]\n{placeholder}\n[resume end]"
    "\n\nReferring to the skills listed in the system message, what skills does this candidate have according to the resume? "
    "Return output as a paragraph"
)

RETURN_FORMAT_ASSISTANT_MESSAGE = (
    "Return the next assistant message in JSON format with three key-value pairs"
    "\nFirst item is the answer and its value is a string representing whether the resume is a good fit, the accepted values are 'no fit', 'potential fit', 'good fit'"
    "\nSecond item is the justification for previous item, and its value is a free text string with up to 50 tokens"
    "\nThird item is the confidence score, and its value is a float between 0 and 1"
    '\n\nExample:\n{"answer":"potential fit","justification":"reason why the resume is a potential fit", "confidence":0.5}'
)

FIT_ASSESSMENT_USER_MESSAGE = (
    "Definitions:"
    "\ngood fit means the resume of the candidate meets almost all requirements stated in the job description, and can be expected to perform well in the role"
    "\npotential fit means the resume of the candidate meets more than half of the requirements stated in the job description, and has the potential to close the gap without significant barrier"
    "\nno fit means the resume of the candidate meets little to up to half of the requirements state in the job description, significant amount of effort will be required for the candidate to close the gap"
    "\n\nInstructions: "
    "\nBased on the skills in the resume and the skills specified in the job description, assess if this candidate is a good fit for the role."
    "Please provided answer for goodness of fit, justification and confidence score for your answer in JSON format described in the last assistant message."
)

RESPONSE_CLEAN_USER_MESSAGE = (
    "The string to be cleaned is:",
    "\n\n{placeholder}"
    "\n\nPlease return output in the JSON format described in the system message."
)


def validate_answer(output_str):
    
    if check_output_is_json(output_str):
        tmp_output_dict = json.loads(output_str)
        return all([
            check_answer_item(tmp_output_dict),
            check_confidence_item(tmp_output_dict), 
            check_justification_item(tmp_output_dict),
        ]) 
    else:
        return False

def check_output_is_json(output_str:str):
    try:
        output_dict = json.loads(output_str)
        return True
    except Exception:
        return False

def check_answer_item(output_dict:Dict):
    if "answer" in output_dict.keys():
        if str(output_dict["answer"]).lower() in ["no fit", "potential fit", "good fit"]:
            return True
        else:
            return False
    else:
        return False

def check_confidence_item(output_dict:Dict):
    if "confidence" in output_dict.keys():
        try:
            if 0 <= float(output_dict["confidence"]) and float(output_dict["confidence"]) <= 1:
                return True
            else:
                return False
        except:
            return False
    else:
        return False

def check_justification_item(output_dict:Dict):
    if "justification" in output_dict.keys():
        return True 
    else:
        return False


class ResumeMatcher():

    def __init__(self, open_ai_client:OpenAI, job_desc:str, resume:str) -> None:
        self.client = open_ai_client
        self.model = "gpt-4o-mini"
        self._messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_MESSAGE
                    }
                ]
            },
        ]   
        self.job_desc = job_desc
        self.resume = resume
        self.required_skills = None
        self.resume_skills = None
    
    # utilities 
    def _query_model(self):
        """Query model with current parameters and messages"""
        response = client.chat.completions.create(
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
    
    def _user_msg(self, template:str, parameterisation:dict=None):
        if parameterisation is not None:
            user_msg = template.format(**parameterisation)
        else:
            user_msg = template
        msg = {
                "role": "user",
                "content": [{"type": "text", "text": user_msg}]
            }
        return msg

    def _assistant_msg(self, text:str):
        msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": text}]
            }
        return msg

    # method
    def check_fit(self) -> Dict:
        
        # grab skills from resume 
        self._messages.append(self._user_msg(template=JOB_DESC_USER_INPUT_TEMPLATE,parameterisation={"placeholder":self.job_desc}))
        job_summ_response = self._query_model()
        self.required_skills = job_summ_response
        self._messages.append(self._assistant_msg(text=job_summ_response))
        
        # grab skills of the person 
        self._messages.append(self._user_msg(template=RESUME_SKILLS_USER_INPUT_TEMPLATE, parameterisation={"placeholder":self.resume}))
        resume_summ_response = self._query_model()
        self.resume_skills = resume_summ_response
        self._messages.append(self._assistant_msg(text=resume_summ_response))

        # assess goodness of fit 
        self._messages.append(self._assistant_msg(text=RETURN_FORMAT_ASSISTANT_MESSAGE))
        self._messages.append(self._user_msg(template=FIT_ASSESSMENT_USER_MESSAGE))
        match_response = self._query_model()

        if validate_answer(match_response):
            return json.loads(match_response)
        else:
            return clean_response(self.client, match_response, 1)


def clean_response(open_ai_client:OpenAI, proposed_response:str, attempts:int=1) -> None:
    
    MAX_ATTEMPTS = 3 
    if attempts > MAX_ATTEMPTS:
        return {"answer":None, "justification":None, "confidence":None}
    
    model = "gpt-4o-mini"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": RETURN_FORMAT_ASSISTANT_MESSAGE
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": str(RESPONSE_CLEAN_USER_MESSAGE).format(placeholder=proposed_response)
                }
            ]
        }
    ]   

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    
    output_str = response.choices[0].message.content

    if validate_answer(output_str=output_str):
        return json.loads(output_str)
    else:
        return clean_response(open_ai_client, output_str, attempts+1)
    


if __name__ == "__main__":
    dev_exmaples = pd.read_csv("data/golden_example.csv")
    print(dev_exmaples.head())
    
    client = OpenAI()

    example_job_desc = dev_exmaples["job_desc"][0]
    example_res = dev_exmaples["resume"][0]

    print("*"*80)
    print(example_job_desc)
    print("\n"+"*"*80)
    print(example_res)

    matcher = ResumeMatcher(client, example_job_desc, example_res)
    print("\n"+"*"*80)
    output = matcher.check_fit()
    print("\n"+"*"*80)
    print(matcher.required_skills)
    print("\n"+"*"*80)
    print(matcher.resume_skills)
    print("\n"+"*"*80)
    pprint(output)
    