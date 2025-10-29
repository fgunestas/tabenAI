import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tqdm
import hashlib
from langchain_core.documents import Document

CSV_PATH=r'C:\Users\fig\PycharmProjects\tabenAI\data\besiktas_reviews_serpapi_part_full.csv'
COLLECTION_NAME = "reviews"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_PERSIST_DIRECTORY = "./chroma_store"



def doc_id(restaurant, review): #1 review 1 id for managing dublicate reviews.
    return hashlib.sha1(f"{restaurant}|{review}".encode("utf-8")).hexdigest()

def normalize_latlon(df):
    cols = {c.lower(): c for c in df.columns}
    # olası isimler: x/y, lon/lat, longitude/latitude
    lat_candidates = ["lat","latitude","y"]
    lon_candidates = ["lon","long","longitude","x"]
    lat_col = next((cols[c] for c in lat_candidates if c in cols), None)
    lon_col = next((cols[c] for c in lon_candidates if c in cols), None)
    return lat_col, lon_col

def build_text(rname, review):
    return f"[{rname}]: {review}"
def db_def():
    hf_ef = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        persist_directory=DB_PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_function=hf_ef
    )
    return db


def main():

    db=db_def()

    df = pd.read_csv(CSV_PATH).fillna("")
    df.columns = df.columns.str.lower()



    lat_col, lon_col = normalize_latlon(df)


    batch = 64
    docs, metas, ids = [], [], []

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        rname = str(row["name"]).strip()
        text = str(row["review"]).strip()

        meta = {"restaurant": rname}
        if "latitude" in row and "longitude" in row:
            try:
                meta["lat"] = float(row["latitude"])
                meta["lon"] = float(row["longitude"])
            except Exception:
                pass

        # LangChain Document objesi
        docs.append(Document(page_content=f"[{rname}]: {text}", metadata=meta))

        if len(docs) >= batch:
            db.add_documents(docs)
            docs = []

    if docs:
        db.add_documents(docs)
    return db

if __name__=="__main__":
    main()

