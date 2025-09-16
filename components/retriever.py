from langchain_community.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import chromadb



def Retriever(query_text: str, top_k: int = 5):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "chroma_store")) #local db path

    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    EMBED_MODEL = "sentence-transformers/paragprase-multilingual-MiniLM-L12-v2"
    hf_ef = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        client=chroma_client,
        collection_name="reviews",  # current collection name
        embedding_function=hf_ef
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})


    return retriever
