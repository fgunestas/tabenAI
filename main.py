import sys,os
import subprocess




def main():
    if os.path.exists("./components/chroma_store/chroma.sqlite3"):
        print("db already exist")
    else:
        subprocess.run(["python", "components/vector_store.py"], check=True)





if __name__ == "__main__":
    main()