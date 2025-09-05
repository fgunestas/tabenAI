import sys,os
import subprocess
from components.retriever import Retriever
import os, chromadb



def main():
    if os.path.exists("./chroma_store/chroma.sqlite3"):
        print("db already exist")
    else:
        subprocess.run(["python", "components/vector_store.py"], check=True)

    res=Retriever("lahmacun", 5)
    print(res)



if __name__ == "__main__":
    main()