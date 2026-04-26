"""
Demonstrates a weather agent with tools, context injection, structured response formatting,
and conversation memory using LangChain with a local Ollama model.
"""

import requests
from langchain.agents import create_agent #main import for creating the agent
from langchain.tools import tool, ToolRuntime # make the functions a tool
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Context_:
    user_id: str

@dataclass
class ResponseFormat_:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

# return_direct -> return the output directly to the user
@tool('get_weather', description="Return weather information for a given city!", return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()



@tool('locate_user', description="Look up a user's city based on their user_id in the context. Call this when the user asks about weather but doesn't specify a city.")
def locate_user(runtime):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Karlsruhe'
        case 'XYZ456':
            return 'Istanbul'
        case "JKLM111":
            return 'Vienna'
        case _:
            return 'Unknown'

#Create free local model :')
model = ChatOllama(
    model="llama3.1:8b",
    temperature=0.3 # Temperature controls how random vs. predictable the model's responses are.
)

checkpointer = InMemorySaver() # this is for remembering conversation

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt = "You are a helpful weather assistant who always cracks jokes and is humorous while remaining helpful.",
    context_schema = Context_,
    response_format= ResponseFormat_,
    checkpointer=checkpointer
)

config = {'configurable': {'thread_id': 1}}

response1 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What about Stuttgart?"}
        ]
    },
    config = config,
    context = Context_(user_id='ABC123'),
)
# Print only the text message content
print(response1['messages'][-1].content)

config = {'configurable': {'thread_id': 2}}

print("Hey!")
response2 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Stuttgart?"}
        ]
    },
    config = config,
    context = Context_(user_id='ABC123'),
)
print(response2['messages'][-1].content)
