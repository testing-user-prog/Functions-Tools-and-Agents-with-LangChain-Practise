from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
api_key=config.get("OPENAI_API_KEY")
model= ChatGroq(
    api_key=api_key, 
    model="llama-3.1-8b-instant"
)


#TASK1: Simple Language translation chain


# # Urdu to English Translator
# template="""Translate the following English sentence to Urdu: "{sentence}" 

# ### RULES:
# 1. Only output a single translated version.
# 2. Do not output anything else other than the translated sentence.
# """
# prompt=ChatPromptTemplate.from_template(template)
# question_dict={
#     "sentence":"I am having breakfast."
# }
# output_parser=StrOutputParser()
# chain= prompt | model | output_parser
# print(chain.invoke(question_dict))

# TASK2: Topic as the input and explanation for it
# output_parser=StrOutputParser()
# template = """ Give me a short brief explanation on {topic} 

# ###NOTE:
# 1. The explanation should be of 4-5 lines.
# 2. Do not output anything else other than the explanatory paragraph. 
# """
# prompt=ChatPromptTemplate.from_template(template)
# chain = prompt | model | output_parser 
# # print(chain.invoke({"topic":"Photosynthesis"}))

# TASK3: Generate MCQS of the particular topic with the help of the LLM and then in a seperate chain ask the LLM to solve the question

output_parser=StrOutputParser()
template1="""Make 4 multiple choice questions on this topic: {topic}

###NOTE:
1. Each multiple choice question should have only 4 options, labelled as a,b,c,d.
2. The questions should be labelled in numbers (1, 2, 3 and so on)
3. Only output the multiple choice questions in the explained format and nothing else
"""

prompt1=ChatPromptTemplate.from_template(template1)

template2="""Here are some of the multiple choice questions, give me the correct answers for them: {questions}

###NOTE:
1. For each of the question only output the question along with all of its options followed by the statement "The correct option is " before displaying the correct option.
2. Do not output anything else other than the desired format.
"""

prompt2=ChatPromptTemplate.from_template(template2)
chain={"questions": prompt1 | model | output_parser} | prompt2 | model | output_parser
print(chain.invoke({"topic":"Retrieval-Augmented Generation"}))




