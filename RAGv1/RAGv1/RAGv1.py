
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import os

# Load environment variables from .env file
load_dotenv()

# GitHub Models uses the OpenAI SDK with a custom endpoint
# Requires GITHUB_TOKEN environment variable to be set
client = OpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com"
)

# 1. Create embeddings
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# 2. Create vector store
documents = ["RAG stands for Retrieval-Augmented Generation.",
             "Embeddings convert text into vectors.",
             "Turtles are unrelated to the RAG model, but they are very cute, aren't they?"]

embeddings = np.vstack([embed(doc) for doc in documents])
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

# 3. Query
query = input("Query: ")
query_embedding = embed(query)

_, I = index.search(np.array([query_embedding]), k=2)
retrieved_docs = [documents[i] for i in I[0]]

# 4. Build prompt
context = "\n".join(retrieved_docs)

response = client.chat.completions.create(
    model="gpt-4o-mini",  # You can also use: Claude-3.5-sonnet, Claude-3-haiku
    messages=[
        {"role": "system", "content": "Answer using the context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print(response.choices[0].message.content)
