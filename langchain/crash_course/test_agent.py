"""
Basic agent test demonstrating tool usage with a simple weather lookup tool and
a local Ollama LLM for conversational AI.
"""

import requests
from langchain.agents import create_agent #main import for creating the agent
from langchain.tools import tool # make the functions a tool
from langchain_ollama import ChatOllama

# return_direct -> return the output directly to the user
@tool('get_weather', description="Return weather information for a given city!", return_direct=True)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

model = ChatOllama(
    model="llama3.1:8b",
    temperature=0.0 # Temperature controls how random vs. predictable the model’s responses are.
)

agent = create_agent(
    model = model,
    tools = [get_weather],
    system_prompt = "You are a helpful weather assistant, who always cracks jokes and is humorous while remaining helpful."
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Waht is the weather like in Karlsruhe? simply describe!"}
    ]
})
print("FULL RESPONSE: ",response)
print(response['messages'][-1].content)
