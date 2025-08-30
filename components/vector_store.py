import pandas as pd
from chromadb.utils import embedding_functions


csv_path=r'C:\Users\fig\PycharmProjects\tabenAI\data\Restaurant_reviews.csv'
EMBED_MODEL = "intfloat/multilingual-e5-large"



ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
)
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef,
    metadata={"hnsw:space": "cosine"},
)

df = pd.read_csv(r'C:\Users\fig\PycharmProjects\tabenAI\data\Restaurant_reviews.csv')
