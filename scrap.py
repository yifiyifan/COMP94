from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained Flan-T5 model and tokenizer
model_name = 'google/flan-t5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the passage, question, and choices
passage = """Position Summary:As a Fox Factory Senior Tax Accountant, you will be supporting the Tax Department regarding direct and in-direct tax activities. This role will be involved in corporate tax strategy, tax provision preparation, and non-income tax activities. You will be responsible for accurate delivery of financial information, non-income tax reporting, tax research, tax planning and preparation of supporting workpapers on various ad-hoc projects. You will work alongside Company Executives, Senior Management, Tax Analyst and the Tax Manager as a member of the corporate tax team. Additionally, you will have an opportunity to learn new skillsets, be involved in the Fox community, and build lasting relationships with members of the Fox Factory Team.
Position Responsibilities:Assist in preparation and review of quarterly and annual ASC 740 calculation for global consolidated group.Support the preparation and assist in the review of the Tax section of SEC Filings, including reports on Form 10-Q and 8-K.Lead transaction tax (sales/use, business & occupation, gross receipts, and property tax) return cycle for monthly, quarterly and annual filings, and facilitate filing of returns on a timely and accurate basis.Sales and Use Tax Compliance and Property Tax Compliance, specifically supporting the preparation of data, filing returns, reviewing work prepared by Tax Analyst, external service provider, and/or Avalara software.Lead centralization of Sales and Use and Property Tax Filings to Fox Factory HQ using Avalara.Respond to inquiries/notices from the IRS and other taxing authorities.Work with external service providers while building rapport and strong relationships with external service team. Manage and maintain relationships with business partners and auditors including investigating, concluding and responding to questions regarding tax matters.Research and resolve issues that arise in the tax compliance, tax provision, and in-direct tax processes.
Specific Knowledge, Skills or Abilities Required:Excellent analytical, communication (written and verbal) and interpersonal skills.Demonstrated project management skillset including project planning and time management.Experience of working with Avalara Sales and Use Tax Reporting Software.Ability to work efficiently and effectively in a team environment.Strong PC skills and familiarity with Microsoft Office software (Word, Access, Excel, and PowerPoint).Willingness to take ownership and responsibility of work and continually look to improve upon those tasks.Internally motivated to seek out answers, generate ideas, and develop new skills.Reliable, detail-oriented, and interested in the Tax career field.
Competencies:Customer Mindset: Exceptional customer experience is primary focus while performing job duties. Quality is a top priority.Adaptability & Innovation: Proactively and willingly adapts to changing business needs and conditions and presents creative and fresh ideas on how to solve problems, gain efficiencies and improve quality.Relationship Building: Builds constructive working relationships characterized by a high level of inclusion, cooperation and mutual respect. Accountability: Takes personal responsibility for the quality and timeliness of work and strives to exceed requirements.Decision Making and Judgment: Makes timely, informed decisions that take into account the facts, goals, constraints and risks.Talent Development (Self and Others): Displays an ongoing commitment to learning and self-improvement; making an effort to acquire new knowledge or skills associated with job responsibilities. Willingness to work with others and coach/teach in effort to develop and support other employees’ development.
Position Qualifications:
Education:Bachelor’s Degree in Accounting or FinanceCPA Candidate preferred
Experience: 3-5 years of Tax Accounting experience in public accounting or corporate tax department
Work Environment and Physical Requirements:Office or production/manufacturing environment depending on assignmentMay be required to lift 20 lbs. frequentlyMay be required to walk, stand, sit, bend and/or lift for long periods of time.May require vision abilities to validate and enter data on computer. 
Disclaimer: This list does not represent all physical demands. Descriptions are representative of those that must be met by employee to successfully perform the essential functions of the job. Reasonable accommodation may be provided to enable individuals with disabilities to perform the jobs’ essential functions. 
FOX provides equal employment opportunities for all employees and applicants for employment without regard to race, color, ancestry, national origin, gender, gender identity, sexual orientation, marital status, religion, age, physical disability (including HIV and AIDS), mental disability, results of genetic testing, or service in the military, or any other characteristic protected by the laws or regulations of any jurisdiction in which we operate. We base all employment decisions –including recruitment, selection, training, compensation, benefits, discipline, promotions, transfers, layoffs, terminations and social/recreational programs –on merit and the principles of equal employment opportunity."""

question = "What level of experience is described in this job posting?"
choices = [
    "A) Junior role", 
    "B) Senior role", 
    "C) Management role",
]


# Prepare the input in the format expected by T5
input_text = f"question: {question} context: {passage} choices: {' '.join(choices)}"
inputs = tokenizer(input_text, return_tensors='pt')

# Generate the answer
outputs = model.generate(**inputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


question2 = "What is the minimum number of years of experience required in this job post answer as an integer?"
input_text2 = f"question: {question2} context: {passage}"
inputs2 = tokenizer(input_text2, return_tensors='pt')

# Generate the answer
outputs2 = model.generate(**inputs2)
answer2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)

question3 = "What is the job family described in this job post?"
choices3 = ["A) accountant", "B) data analyst", "C) data engineer", "D) financial advisor",   "E) software enigneer", "F) None of the above"]


# Prepare the input in the format expected by T5
input_text3 = f"question: {question3} context: {passage} choices: {' '.join(choices3)}"
inputs3 = tokenizer(input_text3, return_tensors='pt')

# Generate the answer
outputs3 = model.generate(**inputs3)
answer3 = tokenizer.decode(outputs3[0], skip_special_tokens=True)

question4 = "What skills does the job applicant must have for this role? do not include qualification and years of experience."
input_text4 = f"question: {question4} context: {passage}"
inputs4 = tokenizer(input_text4, return_tensors='pt')

# Generate the answer
outputs4 = model.generate(**inputs4)
answer4 = tokenizer.decode(outputs4[0], skip_special_tokens=True)

print(f"Answer: {answer}")
print(f"Years of experience: {answer2}")
print(f"Job family: {answer3}")
print(f"Required skills: {answer4}")

# About 3 seconds to answer each question
# should short circuit 