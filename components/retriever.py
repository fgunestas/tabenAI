import chromadb
from chromadb.utils import embedding_functions
import os







def Retriever(query_text: str, top_k: int = 5):
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "chroma_store"))

    client = chromadb.PersistentClient(path=DB_DIR)

    COLLECTION_NAME = "reviews"
    EMBED_MODEL = "nomic-embed-text"

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name=EMBED_MODEL)

    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ollama_ef)

    return collection.query(query_texts=query_text, n_results=top_k)
