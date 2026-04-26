"""
Middleware = code that runs before/after the AI processes your request.

This file shows: how to intercept and modify requests/responses, add logging,
or transform data as it flows through the agent.
"""
import time
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage, AIMessage

# SAMPLE - Middleware

# Create free local model
model = ChatOllama(
    model="llama3.1:8b",
    temperature=0.3
)

class HooksDemo(AgentMiddleware):

    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        """Called when agent starts processing"""
        self.start_time = time.time()
        print("[MIDDLEWARE] before_agent triggered.")

    def before_model(self, state: AgentState, runtime):
        """Called before LLM processes the request"""
        print("[MIDDLEWARE] before_model triggered.")

    def after_model(self, state: AgentState, runtime):
        """Called after LLM generates response"""
        print("[MIDDLEWARE] after_model triggered.")

    def after_agent(self, state: AgentState, runtime):
        """Called when agent finishes"""
        elapsed = time.time() - self.start_time
        print(f"[MIDDLEWARE] after_agent: {elapsed:.2f} seconds")

# Create agent with middleware
agent = create_agent(
    model=model,
    middleware=[HooksDemo()]
)

# Invoke agent
response = agent.invoke({
    'messages': [
        SystemMessage('You are a helpful assistant.'),
        HumanMessage('What is PCA?')
    ]
})

print("\n" + "="*60)
print("AI RESPONSE:")
print("="*60)
print(response['messages'][-1].content)
