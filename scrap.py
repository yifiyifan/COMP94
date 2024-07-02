from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained Flan-T5 model and tokenizer
model_name = 'google/flan-t5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the passage, question, and choices
passage = """Looking for candidates with 4+ yearsâ€™ experience in data science or product analyst roles with a strong background in data analysis, statistical analysis, modelling and specific experience working with product-specific data.  Note : Need some one who can work on w2. Title: Data Scientist/ Product Analyst  Location: San Francisco, CA  Duration: 6 to 7 Months   Project Overview  Drive analysis for Fitbit product teams focused on Fitbitâ€™s products, apps, and services. Analyses include optimizing user lifetime value, improving usage and retention, evaluating content engagement, and suggesting future service innovations for growth. Deliver effective presentations of findings and recommendations to multiple levels of stakeholders, creating visual displays of quantitative information.   Overall Responsibilities Use causal inferential methods to quantify impact on product deliveries when experimentation is not available. Collaborate with cross-functional stakeholders to formulate and complete full cycle analysis that includes data gathering, analysis, ongoing scaled deliverables and presentations. Help Google focus on key decisions to improve products and services.   Top 3 Daily Responsibilities Deep dive on core product features, user segmentation, and relation to user satisfaction and retention. Conduct data analysis to make business recommendations (e.g. cost-benefit, forecasting, impact analysis). Develop and automate reports, iteratively build and prototype dashboards to provide insights at scale, solving for business priorities.   Mandatory Skills Bachelor's Degree or equivalent experience 4-5 years of experience Business acumen & intuition: Knowledge of structured problem-solving, communicating results, risk (e.g., considers business risks, leverages cross-functional teams) for B2C (i.e., consumer facing) products. Coding/Data extraction: Ability to extract relevant information from reading code in one or more core languages (e.g., Python, C++, SQL) and frameworks and ability to leverage the code as a resource to create work output for users or stakeholders. Experience working with highly unstructured / messy datasets and ability to clean and derive insights. Communication and active listening: Ability to clearly explain stats or domain knowledge to people not familiar with the subject matter or who lack a quant background. This includes the ability to explain reasons in a coherent, logical way that is very easy to understand by all. Leverage communication skills and active listening to manage stakeholders and to set proper technical direction for teams or orgs. Data analysis and synthesis: Ability to analyze information, draw conclusions, generate alternatives and solutions, and evaluate outcomes. This includes the ability to use data to add value to business planning and strategies. Data curation, validation, and cleaning: Ability to extract data and validate raw data to ensure it is valid and reliable and ability to clean data based on validation criteria and prepare for further analyzes. Measurement/Applied analysis: Ability to define and rationalize appropriate metrics, create pipelines and dashboards that tell a story. Ability to measure the success of a given effort. Knowledge of different frameworks/architectures/methods/data analysis, ability to select the appropriate approach for the problem context., understanding of the broader context of generating, securing, and increasing the value of the business and revenue outcomes. Modeling concepts/experimental design: Ability to apply multiple approaches and select the right analysis for the problem. Understanding of the mathematical and statistical concepts underpinning measurement, modeling, and experiments. Knowledge of essential statistical methods used to analyze data (e.g., t-tests, descriptive statistics), ability to identify and conduct appropriate basic statistical analysis to determine the basic parameters of a set of data and solve data-related problems. Product analysis leadership: Ability to interact confidently, clearly, and respectfully with others, especially senior leaders, to present, defend, and clarify concerns or issues regarding an existing product, program, or solution, ability to effectively address difficult questions, handle pushback from a high-level audience, and maintain a professional demeanor while engaging in challenging or sometimes high-pressure situations. In addition to influencing stakeholders, this includes actively managing priorities across stakeholders, teams, and projects. Project scoping, execution and influence: Ability to proactively communicate insights and influence stakeholders and subject matter experts to inform decision-making. Ability to convert and uncover open-ended real world problems within the business context into trackable metrics or a structured analytical/statistical framework, as well as the ability to generate business-related insights from data analysis in a way that is meaningful to the stakeholders. Ability to prepare effective presentations in content and function, and to speak competently to the level of the audience. Ability to identify and debug product issues and user pain-points, including the ability to carry out root cause analysis and quantitatively assess critical user journeys, think about big-picture implications, risks, and opportunities.   Self-Rating Required:  Please rate yourself on a scale of 1-5 (5 being the highest). Please list the # of years of experience you have with that particular skill.(For example: SQL- 4, 6)   Skills  Rating  Years of Experience with Skill  Business acumen & intuition    Coding/Data extraction    Communication and active listening    Data analysis and synthesis    Data curation, validation, and cleaning    Measurement/Applied Analysis    Modeling concepts/experimental design    Product analysis leadership    Project scoping, execution and influence    Provide me below information  Name of the Candidate :Current Location :Current Address :Contact Number :Email ID :Hourly Rate on W2 :Interviews or Offers in Pipeline :Interview Availability :Start Availability :Authorization Status :LinkedIn : 
Warm Regards,
Zainab Saba | Talent Acquisition Specialist â€“ US StaffingC: 201 - 905 â€“ 1674 ; +1 510--296 â€“7488 XTN 8384E: zainab.saba@milestone.tech"""

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

print(f"Answer: {answer}")
print(f"Years of experience: {answer2}")
print(f"Job family: {answer3}")

# About 3 seconds to answer each question
# should short circuit 