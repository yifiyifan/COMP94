from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

ARTICLE_TO_SUMMARIZE = (
    """Job Description:Role: Electrical Engineer
Job Summary:   This is a hybrid position with design work completed mostly from home and on-site work completed at the power plant or substation performing design data gathering, installation, and testing.Occasional travel throughout the United States may be required, up to 25%. Candidates should live within one hour of a commercial airport.Duties and Responsibilities:Protection System Design Work:Â· Perform field data collectionÂ· Develop DC schematic drawings, AC elementary drawings, physical drawings, bill of materials, and relay settingsÂ· Develop & review wiring diagrams, cable/conduit schedules, and front panel layoutsÂ· Develop & review as-built drawingsControl System Programming and Design Work:Â· Develop control system architecture, points list, PLC ladder logic, and HMI screen configurationsÂ· Develop & review interconnect/wiring diagramsÂ· Develop & review as-built drawingsField Install Work:Â· Assist with the development of LOTO requirementsÂ· Assist with project schedulesÂ· Support electricians during installsÂ· Perform functional testing and startup checksÂ· Assist with daily updatesÂ· Make project redlinesOther:Â· Attend design review meetingsSkills and Abilities Required:Â· Strong working knowledge of power generation and transmission (G&T) equipment, metering and controls, instrument transformer circuits, relays and relay settings/logic, generator excitation systems, transformers, G&T protective principles and functionality, and testing procedures for G&T equipment is preferredÂ· Experience installing and commissioning equipment in power plants and substations is preferredÂ· Ability to develop and execute commissioning work plans required for a project is preferredÂ· Must be a self-motivated engineer with a strong work ethic who is equally comfortable in the field or at a computerÂ· Must possess strong communication skills and be able to work day-to-day with electriciansÂ· Strong computer skills in all Microsoft platforms, AutoCAD, and MathCADEducation and Experience Requirements:Â· Bachelorâ€™s degree in Electrical Engineering (BSEE)Â· 3+ years of recent and relevant experience with power plant or substation electrical design, installation, and commissioning 
"""
)

context = f"question: What is the minimum number of years of experience required in this job post answer as an integer? context: {ARTICLE_TO_SUMMARIZE}"
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt", truncation=True)

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=10)
# output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(answer)