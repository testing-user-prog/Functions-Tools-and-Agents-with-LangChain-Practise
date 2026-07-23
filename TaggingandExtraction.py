from langchain_core.prompts import ChatPromptTemplate
from urllib3 import response
import requests
import json
from typing import List,Optional
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel,Field
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import  convert_to_openai_function
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
class Person(BaseModel):
  """Information about a person"""

  name: str = Field(description="The full name of the person")
  age: Optional[int] = Field(
      default=None, description="The age of the person if explicitly mentioned"
  )
  gender: Optional[int] = Field(
      default=None, description="The gender of the person"
  )
class Information(BaseModel):
  """Extracted collection of people information from the document"""

  people: List[Person] = Field(
      description="A list of all people mentioned in the text"
  )


with open('config.json', 'r') as config_file:
    config = json.load(config_file)
api_key=config.get("OPENAI_API_KEY")
model= ChatGroq(
    api_key=api_key, 
    model="llama-3.1-8b-instant"
)
functions=[convert_to_openai_function(Information)]
model.bind(functions=functions,function_call={"name":"Information"})
loader=WebBaseLoader("https://www.gutenberg.org/files/11/11-0.txt")
pages=loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200,  
    length_function=len,
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "An article will be passed to you. Extract from it all the people"
            " details that are mentioned in it.\n\nNOTE: In case of no details,"
            " do not assume the answer."
        ),
    ),
    ("user", "{input}"),
])



extraction_chain= prompt | model | JsonOutputFunctionsParser(key='people')




prep=RunnableLambda(
    lambda x:[{"input":doc}for doc in text_splitter.split_documents(x)]

)

def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list

parallel_chain= prep | extraction_chain.map() | flatten 

print(parallel_chain.invoke(pages))

print('program executed')