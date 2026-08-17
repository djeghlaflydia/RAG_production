import os
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
    )

from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing.")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print("document content:")
            print(doc)
            print(doc.page_content)

    finally:
        os.remove(temp_file_path)


def pdf_loader(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"document {i+1} content preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    pdf_loader("./docs/langchain_demo.pdf")

