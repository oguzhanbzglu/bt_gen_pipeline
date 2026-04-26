"""
Middleware = code that runs before/after the AI processes your request.

This file shows: how to intercept and modify requests/responses, add logging,
or transform data as it flows through the agent.
"""

from dataclasses import dataclass
from langchain.agents import create_agent #main import for creating the agent
from langchain_ollama import ChatOllama
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

# SAMPLE - Middleware

#Create free local model :')
model = ChatOllama(
    model="llama3.1:8b",
    temperature=0.3 # Temperature controls how random vs. predictable the model's responses are.
)

@dataclass
class Context_:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
     user_role = request.runtime.context.user_role

     base_prompt = 'You are a helpful and very concise assistant.'

     match user_role:
          case 'expert':
               return f'{base_prompt} Provide detail technical responses.'
          case 'beginner':
               return f'{base_prompt} Keep your explanations simple and basic/'
          case 'child':
               return f'{base_prompt} Explain everything as if you were literally talking to a five-years old.'
          case _:
               return base_prompt

agent = create_agent(
    model = model,
    middleware = [user_role_prompt],
    context_schema = Context_
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Explain PCA."}]
    },
    context=Context_(user_role='expert'))

# print("FULL RESPONSE: ",response)
print(response['messages'][-1].content)
