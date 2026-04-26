"""
RAG = Retrieval Augmented Generation - AI searches your documents to answer questions.

This file shows: the complete RAG pipeline with a retriever tool that the agent can call
to search knowledge base before answering.
"""

import requests
from langchain.agents import create_agent #main import for creating the agent
from langchain_ollama import ChatOllama
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import OllamaEmbeddings

# SAMPLE - RAG (Retrieval Augmented Generation)

#Create free local model :')
model = ChatOllama(
    model="llama3.1:8b",
    temperature=0.9 # Temperature controls how random vs. predictable the model's responses are.
)

embeddings = OllamaEmbeddings(model='nomic-embed-text')

texts = [
    'I love apples.',
    'I think banana is great.',
    'I enjoy oranges.',
    'I like pears.',
    'I dislike mangos',
    'I hate cherry.',
    'I despise raspberries.',
    'I love Linux.',
    'I hate Windows.',
    'I am not fan of Strawberry.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)



# Queery 1
query1 = 'What fruits does the person like?'
results1 = vector_store.similarity_search_with_score(query1, k=3)
for i, (doc, score) in enumerate(results1, 1):
    print(f"  {i}. [{score:.4f}] {doc.page_content}")
print("="*60)

# Queery 2
query2 = 'What fruits does the person hate?'
results2 = vector_store.similarity_search_with_score(query2, k=3)
for i, (doc, score) in enumerate(results2, 1):
    print(f"  {i}. [{score:.4f}] {doc.page_content}")
print("="*60)


retriever = vector_store.as_retriever(search_kwargs={'k':3})

retriever_tool = create_retriever_tool(
    retriever,
    name="k_search",
    description="Search the small product / fruit knowledge base (kb) for information.")

agent = create_agent(
    model = model,
    tools = [retriever_tool],
    system_prompt = (
        "You are a helpful assistan. For questions about Macs, apples, or laptops,"
        "First call the kb_search tool to retrieve context, then answer succinctly. Maybe you have to use it multiple times before answering."
    )
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}
    ]
})
print("FULL RESPONSE: ",response)
print(response['messages'][-1].content)
