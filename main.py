import sys,os
import subprocess
#from retriever import Retriever
import os, chromadb



def main():
    if os.path.exists("./chroma_store/chroma.sqlite3"):
        print("db already exist")
    else:
        subprocess.run(["python", "components/vector_store.py"], check=True)

    #prompt = input("Related restaurant reviews").strip()
    #results = Retriever(query_text=prompt, top_k=10)
    #print(results)
    from chromadb.utils import embedding_functions

    DB_DIR = os.path.abspath("./chroma_store")
    COLL = "reviews"

    client = chromadb.PersistentClient(path=DB_DIR)

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")
    col = client.get_or_create_collection(name=COLL, embedding_function=ollama_ef)

    res = col.query(query_texts=["lahmacun"], n_results=5)

    print(res["documents"][0])  # en iyi 5 snippet
    print(res["metadatas"][0])  # onların metadata'ları




if __name__ == "__main__":
    main()