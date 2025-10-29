import subprocess
from components.rag_pipeline import rag_pipeline, rest_prep
from components.vector_store import main as build_database, db_def
from components.add_review import add_new_review
import os



def main():
    if os.path.exists("./chroma_store/chroma.sqlite3"):
        print("db already exist")
    else:
        build_database()

    db=db_def()
    #rag_pipeline(db,"Beyoğlunda tavuk yiyebileceğim yerler.")
    add_new_review(db,"deneme","slm")

if __name__ == "__main__":
    main()