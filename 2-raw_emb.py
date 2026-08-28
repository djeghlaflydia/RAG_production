# Embedding = transformer du texte en représentation numérique permettant de comparer sa similarité sémantique.

'''
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client =OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# 1. Generate a response
# =========================

CONVERSATION = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"system","content":"you are a helpful assistant."},
        {"role":"user","content":"what is the capital of France?"},
    ]
)

print(CONVERSATION.choices[0].message.content)

# =========================
# 2. Generate embeddings
# =========================

response = client.embeddings.create(
    input="your text string goes here",
    model="text-embedding-3-small")
print(response)
'''

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# =========================
# 1. Generate a response
# =========================

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="What is the capital of France?"
)

print(response.text)


# =========================
# 2. Generate embeddings
# =========================

embedding_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="your text string goes here"
)

embedding = embedding_response.embeddings[0].values

print("Number of values:", len(embedding))
print("First 5 values:", embedding[:5])