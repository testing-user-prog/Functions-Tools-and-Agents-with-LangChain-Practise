# Error handling using pydantic
from pydantic import BaseModel,Field
from langchain_core.utils.function_calling import convert_to_openai_function as convert_pydantic_to_openai_function
from langchain_core.prompts import ChatPromptTemplate
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
api_key=config.get("OPENAI_API_KEY")
model= ChatGroq(
    api_key=api_key, 
    model="llama-3.1-8b-instant"
)
class Employee(BaseModel):
    name:str
    age:int
    employeeid:int
User=Employee(name='Zaid',age='20',employeeid=67)




#Practising the LLM functioning calling using Pydantic and LCEL
class gettime(BaseModel):
    """Call this function to get the time of a particular region"""
    continent: str = Field(description="Continent where the region is located")
    city: str = Field(description="The city whose time you want to get")


class calculatepercentage(BaseModel):
    """Call this function when the user wants to calculate his marks""" 
    totalmarks: int = Field(description="Total marks of the exam")
    obtainedmarks: int = Field(description="The marks that are being earned by the user")

functions=[convert_pydantic_to_openai_function(gettime),convert_pydantic_to_openai_function(calculatepercentage)]
new_model=model.bind(functions=functions)
prompt = ChatPromptTemplate.from_messages([
    ("user", "{question}")
])



chain=  prompt | new_model 
print(chain.invoke({"question": "I recieved 785 marks out of 800 in my maths test"}))
