from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import wget
import os

# Downloading file from the Source
file_name = "company_policy.txt"
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt"

if not os.path.exists(file_name):
    wget.download(url, out=file_name)
    print("\nDocument downloaded\n")
else:
    print("\nDocument already exists\n")


# using Textloader to load a document
loader = TextLoader(file_name)
docs = loader.load()


# Where Splitter is created which then used for chunks
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")


# Embeddings model Initialize
embedding_model = OllamaEmbeddings(model="nomic-embed-text")


# Storing Embeddings to Chroma
db = Chroma.from_documents(
    chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("\nDocuments embedded and stored in Chroma DB\n")