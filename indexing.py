from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

CHROMA_PATH = "chromadb"
DOCUMENTS_PATH = "data"


def load_documents(path):
    reader = PyPDFDirectoryLoader(path=path)
    docs = reader.load()

    return docs


def split_documents(docs):
    char_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunks = char_split.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(docs_chunks)} chunks.")

    return docs_chunks


def store_documents(chunks):
    # create db from chunks of documents
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
        ),
        persist_directory=CHROMA_PATH,
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return vector_store


def indexing(path):
    docs = load_documents(path)
    chunks = split_documents(docs)
    vector_store = store_documents(chunks)

    return vector_store


if __name__ == "__main__":
    indexing(path=DOCUMENTS_PATH)
