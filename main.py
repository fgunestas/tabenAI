import subprocess
from components.rag_pipeline import rag_pipeline, rest_prep
import os



def main():
    if os.path.exists("./chroma_store/chroma.sqlite3"):
        print("db already exist")
    else:
        subprocess.run(["python", "components/vector_store.py"], check=True)

    rag_pipeline("Beşiktaş’ta iyi lahmacun yapan yerler hangileri?")


if __name__ == "__main__":
    main()