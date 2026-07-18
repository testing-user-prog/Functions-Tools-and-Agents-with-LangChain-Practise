
import requests
import json
from datetime import datetime
from openai import OpenAI
# this function returns the data_time provided the continent and the name of the City is entered
def gettime(continent,city):
    response=requests.get(f'https://time.now/developer/api/timezone/{continent}/{city}')
    date_obj =datetime.fromisoformat(json.loads(response.content)['datetime'])
    return f'{date_obj.hour}:{date_obj.minute}'
# print(gettime('Asia','Karachi'))
with open('config.json', 'r') as config_file:
    config = json.load(config_file)
api_key=config.get("OPENAI_API_KEY")
client = OpenAI(
    api_key=api_key, 
    base_url="https://api.groq.com/openai/v1"
)
functions=[
    {
        "name":"gettime",
        "description":"Get the current time of a given location",
        "parameters": {
            "type" : "object",
            "properties" :
            {
                "continent" : {
                    "type":"string",
                    "description":"The continent in which the location exists e.g. Asia, Africa"
                },
                "city":
                {
                    "type":"string",
                    "description":"The city of the location e.g Karachi, London"

                }

            },
            "required":["continent","city"] 

        }
        

    }

]
messages=[
    {
        "role":"user",
        "content":"What's the time in Karachi?"
    }
]
response=client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    functions=functions
)
print(response)
messages.append(response.choices[0].message)
args=json.loads(response.choices[0].message.tool_calls[0].function.arguments)
result=gettime(**args)
messages.append(
    {
        "role":"function",
        "content":f'{result}',
        "name": "gettime"

    }
)
response=client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages
)
print(response)
