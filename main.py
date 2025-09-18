import subprocess
from components.rag_pipeline import rag_pipeline
import os



def main():
    if os.path.exists("./chroma_store/chroma.sqlite3"):
        print("db already exist")
    else:
        subprocess.run(["python", "components/vector_store.py"], check=True)

    rag_pipeline("asd")


if __name__ == "__main__":
    main()