
from openai import OpenAI
import faiss
import numpy as np

client = OpenAI()

# 1. Create embeddings
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# 2. Create vector store
documents = ["RAG stands for Retrieval-Augmented Generation.",
             "Embeddings convert text into vectors."]

embeddings = np.vstack([embed(doc) for doc in documents])
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

# 3. Query
query = "What is RAG?"
query_embedding = embed(query)

_, I = index.search(np.array([query_embedding]), k=2)
retrieved_docs = [documents[i] for i in I[0]]

# 4. Build prompt
context = "\n".join(retrieved_docs)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer using the context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print(response.choices[0].message.content)