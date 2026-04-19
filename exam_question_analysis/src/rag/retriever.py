import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../../data/chroma_db")

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def retrieve_context(query: str) -> str:
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant pedagogy guidelines found."
    return "\n\n".join([doc.page_content for doc in docs])