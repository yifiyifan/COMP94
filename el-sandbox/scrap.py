from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained Flan-T5 model and tokenizer
model_name = 'google/flan-t5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")

# Define the passage, question, and choices

# long example - 1300+ token
# passage = """Sr. Electrical Engineer System Integrator â€“ EV Marine Propulsion
# This position is responsible for the strategy and implementation of software specific to high voltage EV marine propulsion systems, auxiliary power distribution, HV safety system, user interface, and overall development of the vehicle supervisory control system. The System Integrator is expected to move at an aggressive pace within the development teams agile frameworkâ€“ keeping track of program milestones, deliverables and pulling together the contributions into a cohesive final product. The Integration Engineer is also expected to get their hands dirty in the details of their sub-systems â€“ digging through source code, root causing issues and proposing solutions to complex problems, as well as testing prototype firmware implementations.The ideal candidate has experience designing and analyzing control systems, a demonstrated track record of proactive communication and development team leadership, a demonstrated ability to innovate and possesses practical knowledge of electrical, mechatronic and software design.ResponsibilitiesLead software development for the Forza X1 product familyProvide technical leadership to the engineering development teamPromote and champion software engineering best practicesInterface with multiple stakeholders to gather appropriate requirements for feature updatesÂ· Work in the design, selection, analysis, system integration, and testing of marine EV systems, sensors, actuators, controllers, pneumatic and hydraulic systems, and electromechanical systems, including conceptual design, detailed design, testing, validation, and product implementation.Ensure the relevant subsystems are designed with appropriate requirements, interfaces, and interactions to achieve specific functionality.Work with manufacturing to ensure the function(s) are rolled out and functioning properly. This can include development of processes, diagnostic tools and methods, and root-cause diagnosis for product issues.Collaborate and validate the functions, interfaces, and interactions of the system. Identify areas of concern. Extensive knowledge in API / Database design and development best practicesReview code as a part of source control, code reviews, and security checksCommunicate effectively with peers and key non-technical stakeholdersÂ· Work with all appropriate hardware and software protocols, vendors, and systems to understand, troubleshoot, and control program control systems. The systems may include, but may not be limited to: C, C++, Reach, CANBUS, J1939, proprietary motor control and battery management systems, Garmin, Garmin One-helm, HTML5, Javascript, NMEA 2000 and others.
# RequirementsBSc, MSc or PhD in a relevant engineering disciplineEstablished background in EV development, EV Charging or Power Electronic Systems in the automotive or comparable industry, with a combination of theoretical and practical engineering experienceDemonstrated knowledge of foundational engineering topics (e.g., mechanics, physics, metallurgy, manufacturing methods, electronics, software, control systems) within job-related areaExperience in automotive diagnostics development is strongly preferredProven success in driving service and field requirements in a complex, design-focused environmentStrong analytical and structured problem-solving capabilities and a knack for tackling problems that are unusual and complexMinimum of 5 years working on development of complex electro-mechanical integration projects. Exceptional ability to keep multiple projects moving forwards in parallel.Hands-on technical experience debugging complex systems involving networked microprocessors and software-controlled electrical or electromechanical devices.Proven ability to use programming to solve challenging problems and increase own/teamâ€™s efficiency through automation.Experience with C++ or Python are preferred.Ability to fluently interpret system data to gain a full understanding of logged events.Thorough understanding of mechanics fundamentals - and ability to apply them to automotive concepts.Understanding of low voltage and high voltage circuits and how to debug them.Experience with CAN and Vector CAN tools is a plus.Experience with Failure Mode and Effects Analysis (FMEA) and Hazard Analysis and Risk Assessment (HARA) is a plus.Knowledgeable in NMEA 2000 and J1939 protocolsUI/UX Design experience including Figma, Sketch, Adobe XDPassion for boating and watersportsBe part of something amazing!Come work alongside some innovative minds and move the marine industry forward. Beyond providing competitive salaries, weâ€™re providing a community for innovators who want to make an immediate and significant impact. If you are driven to create a better, more sustainable future, then this is the right place for you.At Forza X1 , we donâ€™t just welcome diversity - we celebrate it! Forza X1 is proud to be an equal opportunity workplace. We are committed to equal employment opportunity regardless of race, color, national or ethnic origin, age, religion, disability, sexual orientation, gender, marital status, and any other characteristic protected under applicable State or Federal laws and regulations.Job Types: Full-time, Contract, InternshipSalary: commensurate with experience and abilityBenefits:Employee discountFlexible scheduleHealth insurancePaid time offRelocation assistanceEducation:Bachelor's in associated field of engineering to meet requirements (Preferred)
# Experience:Computer Software, Mechanical or Electrical Engineering: 5-10 years (Preferred)
# Full Time Opportunity:Yes
# Work Location:Remote, then relocation to future factory in the Marion, NC area.
# This Job Is Ideal for Someone Who Is:Dependable -- more reliable than spontaneousAdaptable/flexible -- enjoys doing work that requires frequent shifts in directionDetail-oriented -- would rather focus on the details of work than the bigger pictureAchievement-oriented -- enjoys taking on challenges, even if they might failAutonomous/Independent -- enjoys working with little directionInnovative -- prefers working in unconventional ways or on tasks that require creativityCompany's website:www.forzax1.com
# Work Remotely:Yes, for approximately 18 months, then permanent relocation 
# COVID-19 Precaution(s):Remote interview processPersonal protective equipment provided or requiredSocial distancing guidelines in placeVirtual meetingsSanitizing, disinfecting, or cleaning procedures in place
# """

passage = """Job Description:Role: Electrical Engineer
Job Summary:   This is a hybrid position with design work completed mostly from home and on-site work completed at the power plant or substation performing design data gathering, installation, and testing.Occasional travel throughout the United States may be required, up to 25%. Candidates should live within one hour of a commercial airport.Duties and Responsibilities:Protection System Design Work:Â· Perform field data collectionÂ· Develop DC schematic drawings, AC elementary drawings, physical drawings, bill of materials, and relay settingsÂ· Develop & review wiring diagrams, cable/conduit schedules, and front panel layoutsÂ· Develop & review as-built drawingsControl System Programming and Design Work:Â· Develop control system architecture, points list, PLC ladder logic, and HMI screen configurationsÂ· Develop & review interconnect/wiring diagramsÂ· Develop & review as-built drawingsField Install Work:Â· Assist with the development of LOTO requirementsÂ· Assist with project schedulesÂ· Support electricians during installsÂ· Perform functional testing and startup checksÂ· Assist with daily updatesÂ· Make project redlinesOther:Â· Attend design review meetingsSkills and Abilities Required:Â· Strong working knowledge of power generation and transmission (G&T) equipment, metering and controls, instrument transformer circuits, relays and relay settings/logic, generator excitation systems, transformers, G&T protective principles and functionality, and testing procedures for G&T equipment is preferredÂ· Experience installing and commissioning equipment in power plants and substations is preferredÂ· Ability to develop and execute commissioning work plans required for a project is preferredÂ· Must be a self-motivated engineer with a strong work ethic who is equally comfortable in the field or at a computerÂ· Must possess strong communication skills and be able to work day-to-day with electriciansÂ· Strong computer skills in all Microsoft platforms, AutoCAD, and MathCADEducation and Experience Requirements:Â· Bachelorâ€™s degree in Electrical Engineering (BSEE)Â· 3+ years of recent and relevant experience with power plant or substation electrical design, installation, and commissioning 
"""


choices = [
    "A) junior-role", 
    "B) senior-role", 
]


# Prepare the input in the format expected by T5
context = """junior-level positions: junior-level job title may include "junior", "staff", "intern", "graduate" 

senior-level positions: senior-level job title may include "senior", "lead", "principal".   """ + passage
input_text = f"question: What level of experience is indicated by the responsibility described in this job posting? context: {passage} choices: {' '.join(choices)}"
inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
print(inputs["input_ids"].shape)
# Generate the answer
outputs = model.generate(**inputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


question2 = "What is the minimum number of years of experience required in this job post answer as an integer?"
input_text2 = f"question: {question2} context: {passage}"
inputs2 = tokenizer(input_text2, return_tensors='pt').to("cuda")

# Generate the answer
outputs2 = model.generate(**inputs2)
answer2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)

question3 = "What is the job family described in this job post?"
choices3 = ["A) accountant", "B) data analyst", "C) data engineer", "D) financial advisor",   "E) software enigneer", "F) None of the above"]


# Prepare the input in the format expected by T5
input_text3 = f"question: {question3} context: {passage} choices: {' '.join(choices3)}"
inputs3 = tokenizer(input_text3, return_tensors='pt').to("cuda")

# Generate the answer
outputs3 = model.generate(**inputs3)
answer3 = tokenizer.decode(outputs3[0], skip_special_tokens=True)

question4 = "What skills does the job applicant must have for this role? do not include qualification and years of experience."
input_text4 = f"question: {question4} context: {passage}"
inputs4 = tokenizer(input_text4, return_tensors='pt').to("cuda")

# Generate the answer
outputs4 = model.generate(**inputs4)
answer4 = tokenizer.decode(outputs4[0], skip_special_tokens=True)

print(f"Answer: {answer}")
print(f"Years of experience: {answer2}")
print(f"Job family: {answer3}")
print(f"Required skills: {answer4}")

# About 3 seconds to answer each question
# should short circuit 