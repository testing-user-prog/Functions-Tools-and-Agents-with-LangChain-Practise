from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import json
from langchain_classic.agents import AgentExecutor
from langchain_core.tools import tool
from pydantic import BaseModel,Field
import requests
from datetime import datetime
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_classic.memory import ConversationBufferMemory 
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
api_key=config.get("OPENAI_API_KEY")
model= ChatGroq(
    api_key=api_key, 
    model="llama-3.1-8b-instant"
)
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
memory=ConversationBufferMemory(return_messages=True,memory_key="chat_history")
prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Only use the tools explicitly provided to you. "
        "Do not attempt to call any unlisted tools. If no tool is relevant to the user request, "
        "answer directly using regular text."
    ),
   MessagesPlaceholder(variable_name="agent_scratchpad"),
   ("user","{input}"),
   MessagesPlaceholder(variable_name='chat_history')
])
tools=[gettime]

chain =(
   RunnablePassthrough.assign(
      agent_scratchpad=lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
   )
) | prompt | model.bind_tools(tools) | ToolsAgentOutputParser()
agent_executor=AgentExecutor(agent=chain,tools=tools,verbose=False,memory=memory)


while True :
   question=input("Ask a question: \n")
   response=agent_executor.invoke({"input":question})
   print(response['output'])







