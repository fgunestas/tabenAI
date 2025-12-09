import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import tqdm
import hashlib
from langchain_core.documents import Document
import torch
import os

# Bu dosyanın (vector_store.py) olduğu klasörü bul
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Bir üst klasöre çık (components -> root) ve 'data' klasörüne git
# Docker içinde bu yol otomatik olarak '/app/data/...' olacaktır.
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "besiktas_reviews_serpapi_part_full.csv")

# Yolu mutlak hale getir (garanti olsun)
CSV_PATH = os.path.abspath(CSV_PATH)

COLLECTION_NAME = "reviews"
EMBED_MODEL = "BAAI/bge-m3"
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model yükleniyor... Kullanılan cihaz: {device}")

    hf_ef = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    db = Chroma(
        persist_directory=DB_PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_function=hf_ef
    )
    return db


def main():
    db = db_def()

    df = pd.read_csv(CSV_PATH).fillna("")
    df.columns = df.columns.str.lower()

    batch = 64 # 8GB VRAM ile bu batch size gayet güvenlidir.
    docs, ids = [], [] # ID listesi de tutuyoruz

    print(f"Toplam {len(df)} veri işleniyor...")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        rname = str(row["name"]).strip()
        text = str(row["review"]).strip()

        # ID Oluşturma (Veri tekrarını önlemek için)
        unique_id = doc_id(rname, text)

        meta = {"restaurant": rname}
        if "latitude" in row and "longitude" in row:
            try:
                val_lat = row["latitude"]
                val_lon = row["longitude"]
                if val_lat != "" and val_lon != "":
                    meta["lat"] = float(val_lat)
                    meta["lon"] = float(val_lon)
            except Exception:
                pass

        # LangChain Document objesi
        docs.append(Document(page_content=f"[{rname}]: {text}", metadata=meta))
        ids.append(unique_id)

        if len(docs) >= batch:
            db.add_documents(documents=docs, ids=ids)
            docs, ids = [], []

    if docs:
        db.add_documents(documents=docs, ids=ids)
    return db

if __name__=="__main__":
    main()

