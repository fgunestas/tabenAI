from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import chromadb



def Retriever():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "chroma_store")) #local db path

    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    hf_ef = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        client=chroma_client,
        collection_name="reviews",  # current collection name
        embedding_function=hf_ef
    )

    retriever = db.as_retriever(search_kwargs={"k": 2})


    return retriever
