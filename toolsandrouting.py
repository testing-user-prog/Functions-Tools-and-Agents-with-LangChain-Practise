from datetime import datetime
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
import requests

with open("config.json", "r") as config_file:
  config = json.load(config_file)
api_key = config.get("OPENAI_API_KEY")

model = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")


class getclass_args(BaseModel):
  continent: str = Field(
      description="The name of the continent in which the region is located"
  )
  city: str = Field(description="The city where the region is located")


@tool(args_schema=getclass_args)
def gettime(continent, city):
  """Call this function when you want to get the time of a region"""
  response = requests.get(
      f"https://time.now/developer/api/timezone/{continent}/{city}"
  )
  date_obj = datetime.fromisoformat(json.loads(response.content)["datetime"])
  return f"{date_obj.hour}:{date_obj.minute}"


tools = [gettime]
model = model.bind_tools(tools)
# Generic prompt template that allows passing any instructions or questions dynamically
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful assistant. Only use the provided tools if they"
            " are strictly necessary to answer the user request. Do not"
            " attempt to call tools that have not been provided."
        ),
    ),
    ("user", "{input}"),
])

def parse_output(response):
  if response.content!='':
    return response.content

  if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_map = {t.name: t for t in tools}
    selected_tool = tool_map[tool_call["name"]]
    return selected_tool.invoke(tool_call["args"])
  


chain = prompt | model | parse_output

print(chain.invoke({"input": "What is quadratic formula?"}))