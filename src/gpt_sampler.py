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

# RETURN_FORMAT_ASSISTANT_MESSAGE = (
#     "Return the next assistant message in JSON format with three key-value pairs"
#     "\nFirst item is the answer and its value is a string representing whether the resume is a good fit, the accepted values are 'poor fit', 'potential fit', 'good fit'"
#     "\nSecond item is the justification for previous item, and its value is a free text string with up to 100 tokens"
#     "\nThird item is the confidence score, and its value is a float between 0 and 1"
#     '\n\nExample:\n{"answer":"potential fit","justification":"This candidate meets some requirement of the role but has noticeable gap that needs to be closed to be a strong candidate", "confidence":0.5}'
# )

# FIT_ASSESSMENT_USER_MESSAGE = (
#     "Definitions:"
#     "\n\npoor fit:"
#     "\nA candidate in this category lacks more than one mandatory requirements for the role " 
#     "and / or they also demonstrate limited relevant experience to the role. "
#     "Significant gaps in key areas indicate that they would require extensive training and development to reach the expected competency level for the role. "
#     "These candidates are not suitable for the position at this time."
#     "\n\npotential fit:"
#     "\nCandidates in the potential fit category meet majority of the mandatory technical skills but have some gaps in either mandatory or preferred requirements. "
#     "While they show some relevant experience and proficiency in both technical and non-technical skills, "
#     "these are limited and not deeply aligned with the specific role requirements. "
#     "They may lack some preferred qualifications. "
#     "With further evaluation and potential development, they could become strong contributors to the team."
#     "\n\ngood Fit:"
#     "\nCandidates classified as a good fit if they meet most or all of the mandatory technical skills. "
#     "Meeting some of the preferred requirements should increase the chance that the candidate is a good fit"
#     "Minor gaps in technical skills are acceptable. "
#     "They demonstrate relevant experience and proficiency in both technical and non-technical skills, "
#     "with clear evidence of successful project involvement or contributions. "
#     "These candidates have extensive relevant experience, "
#     "showing clear potential for immediate contribution to the role. "
#     "They are considered strong contenders for the position and are suitable for moving forward in the interview process."
#     # "\n\nHow to distinguish between poor fit, potential fit and good fit:"
#     # "\nIf there are clear indication of good fit. such that we can comfortably proceed this candidate to the next round of interview, prefer good fit over potential fit."
#     # "\nIf there are clear red flags about the candidate, prefer poor fit over potential fit"
#     # "\nReserve potential fit for candidates that you may put on the shortlist"
#     "\n\nPrefer good fit or poor fit over potential fit, reserve potential fit for candidate that you would shortlist but not proceed to next round of interview."
#     "\n\nInstructions: "
#     "\nAssess if this candidate is a good fit for the role based on the discussion above."
#     "Please provided answer for goodness of fit, justification and confidence score for your answer in JSON format described in the last assistant message."
# )

RETURN_FORMAT_ASSISTANT_MESSAGE = (
    "Return the next assistant message in JSON format with three key-value pairs"
    "\nFirst item is the answer and its value is a string representing whether the resume is a good fit, the accepted values are 'poor fit', 'good fit'"
    "\nSecond item is the justification for previous item, and its value is a free text string with up to 100 tokens"
    "\nThird item is the confidence score, and its value is a float between 0 and 1"
    '\n\nExample:\n{"answer":"good fit","justification":"This candidate meets some requirement of the role but has noticeable gap that needs to be closed to be a strong candidate", "confidence":0.5}'
)

FIT_ASSESSMENT_USER_MESSAGE = (
    "Definitions:"
    "\n\npoor fit:"
    "\nA candidate in this category lacks multiple mandatory requirements for the role " 
    "and / or they also demonstrate limited relevant experience to the role. "
    "Significant gaps in key areas indicate that they would require extensive training and development to reach the expected competency level for the role. "
    "These candidates are not suitable for the position at this time."
    "\n\ngood Fit:"
    "\nCandidates classified as a good fit if they meet most or all of the mandatory technical skills. "
    "Meeting some of the preferred requirements should increase the chance that the candidate is a good fit"
    "Minor gaps in technical skills are acceptable. "
    "They demonstrate relevant experience and proficiency in both technical and non-technical skills, "
    "with clear evidence of successful project involvement or contributions. "
    "These candidates have extensive relevant experience, "
    "showing clear potential for immediate contribution to the role. "
    "They are considered strong contenders for the position and are suitable for moving forward in the interview process."
    # "\n\nHow to distinguish between poor fit, potential fit and good fit:"
    # "\nIf there are clear indication of good fit. such that we can comfortably proceed this candidate to the next round of interview, prefer good fit over potential fit."
    # "\nIf there are clear red flags about the candidate, prefer poor fit over potential fit"
    # "\nReserve potential fit for candidates that you may put on the shortlist"
    # "\n\nPrefer good fit or poor fit over potential fit, reserve potential fit for candidate that you would shortlist but not proceed to next round of interview."
    "\n\nInstructions: "
    "\nAssess if this candidate is a good fit for the role based on the discussion above."
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
        # if str(output_dict["answer"]).lower() in ["poor fit", "potential fit", "good fit"]:
        if str(output_dict["answer"]).lower() in ["poor fit", "good fit"]:

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
        return {"answer":None, "justification":None, "confidence":0}
    
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

    response = open_ai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    
    output_str = response.choices[0].message.content

    if validate_answer(output_str=output_str):
        return json.loads(output_str)
    else:
        return clean_response(open_ai_client, output_str, attempts+1)
    
