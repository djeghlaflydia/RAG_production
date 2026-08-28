from langchain_text_splitters import (RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter)
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental import HybridSearch
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import EnsembleRetriever
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

