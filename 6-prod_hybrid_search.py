from langchain_text_splitters import (RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter)
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental import HybridSearch
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from dotenv import load_dotenv

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001")

# Documents with both semantic content AND specific identifiers
documents = [
    Document(
        page_content='Product SKU-7742X is our flagship router. It supports'
        'gigabit speeds and advanced QoS features.',
        metadata={'type':'product'}
    ),
    Document(
        page_content='For network connectivity issues, first check the'
        'enthernet cable and router stauts lights.',
        metadata={'type':'troublesshooting'}
    ),
    Document(
        page_content='Error code E_CONN_REFUSED indicates the server'
        'rejected the connection. check firewall settings.',
        metadata={'type':'error'}
    ),
    Document(
        page_content='The authentication process requires valid credentials.'
        'Use OAuth2 for secure API access.',
        metadata={'type':'authentication'}
    ),
    Document(
        page_content='Router configuration guide: Access the admin panel'
        'at 192.168.1.1 to modify settings',
        metadata={'type':'config'}
    ),
    Document(
        page_content='WCAG 2.1 compliance requires all images to have'
        'alt text and sufficient color contrast.',
        metadata={'type':'compliance'}
    )
]

print (f"Loaded {len(documents)} documents")

# Create embeddings and vector store
vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="hybrid_test",
)

#creat vector retriever
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3} #return top 3
)

print("Vector retriever ready")


# BM25 works on the raw text
bm25_retrievers = BM25Retriever.from_documents(documents, k=3)

print("BM25 retriever ready")


#Combine with EnsembleRetriever to create a hybrid search
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retrievers],
    weights=[0.5, 0.5]  # Adjust weights as needed
)
print("Hybrid retriever ready")
