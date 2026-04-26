"""
RAG = Retrieval Augmented Generation - giving the AI access to your documents/knowledge base
so it can answer questions based on your data, not just its training data.

This file shows: converting text to embeddings (vector numbers), storing them in FAISS database,
and finding similar documents when you search.
"""

import requests
from langchain.agents import create_agent #main import for creating the agent
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import OllamaEmbeddings

# SAMPLE - RAG (Retrieval Augmented Generation)

#Create free local model :')
model = ChatOllama(
    model="llama3.1:8b",
    temperature=0.1 # Temperature controls how random vs. predictable the model's responses are.
)

embeddings = OllamaEmbeddings(model='nomic-embed-text')

texts = [
    'Apple makes very good computers.',
    'I believe Apple is innovative.',
    'I love apples.',
    'I am a fan of MacBooks.',
    'I enjoy oranges.',
    'I like Lenovo Thinkpads.',
    'I think pears taste very good.',
    'Apples and oranges are my favorite fruits'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)



# Queery 1
query1 = 'Apples and oranges are my favorite fruits'
results1 = vector_store.similarity_search_with_score(query1, k=3)
for i, (doc, score) in enumerate(results1, 1):
    print(f"  {i}. [{score:.4f}] {doc.page_content}")
print("="*60)

# Query 2
query2 = 'Which computer brand is best for development?'
results2 = vector_store.similarity_search_with_score(query2, k=3)
for i, (doc, score) in enumerate(results2, 1):
    print(f"  {i}. [{score:.4f}] {doc.page_content}")
print("="*60)

# Query 3
query3 = 'Apple products are great.'
results3 = vector_store.similarity_search_with_score(query2, k=3)
for i, (doc, score) in enumerate(results3, 1):
    print(f"  {i}. [{score:.4f}] {doc.page_content}")
