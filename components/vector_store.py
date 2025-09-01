import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import tqdm
import hashlib

CSV_PATH=r'C:\Users\fig\PycharmProjects\tabenAI\data\besiktas_reviews_serpapi_part_full.csv'
EMBED_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "reviews"


ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
)

client = chromadb.PersistentClient(path="./chroma_store")

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef,
    metadata={"hnsw:space": "cosine"},
)

def doc_id(restaurant, review): #1 review 1 id for managing dublicate reviews.
    h = hashlib.sha1()
    h.update(f"{restaurant}|{review}".encode("utf-8"))
    return h.hexdigest()

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

df = pd.read_csv(CSV_PATH).fillna("")
df.columns = df.columns.str.lower()

print(df.columns)
required = {"name","review"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV en az şu kolonları içermeli: {required}")

lat_col, lon_col = normalize_latlon(df)
if not lat_col or not lon_col:
    # lat/lon yoksa da devam eder; sadece geo-sorting yapamayız
    print("UYARI: lat/lon kolonları bulunamadı; sadece semantik arama yapılır.")
else:
    print(f"Konum kolonları: lat={lat_col}, lon={lon_col}")

batch = 64
docs, metas, ids = [], [], []

for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    rname = str(row["name"]).strip()
    text = str(row["review"]).strip()
    rid = doc_id(rname, text)

    meta = {
        "name": rname,
        "rating": str(row.get("rating","")),
        "source": row.get("source","csv"),
        "city": row.get("city",""),
        "district": row.get("district",""),
    }
    if lat_col and lon_col:
        try:
            meta["lat"] = float(row[lat_col])
            meta["lon"] = float(row[lon_col])
        except Exception:
            pass

    ids.append(rid)
    docs.append(build_text(rname, text))
    metas.append(meta)

    if len(ids) >= batch:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        ids, docs, metas = [], [], []

if ids:
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
