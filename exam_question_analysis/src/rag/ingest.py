import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_DIR = os.path.join(os.path.dirname(__file__), "../../data/pedagogy_docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../../data/chroma_db")

def ingest_documents():
    print("📚 Loading pedagogy documents...")
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("⚠️  No documents found in pedagogy_docs/")
        return

    print(f"✅ Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"✅ Stored in Chroma at {CHROMA_DIR}")

if __name__ == "__main__":
    ingest_documents()