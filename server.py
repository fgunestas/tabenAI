import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import os

# --- 1. Kendi Modüllerimizi Import Ediyoruz ---
from components.add_review import add_new_review
from components.vector_store import (main as build_database, db_def,DB_PERSIST_DIRECTORY)
# RAG pipeline'ımızı alıyoruz
from components.rag_pipeline import rag_pipeline

# --- 2. API Uygulamasını Başlatma ---
app = FastAPI(
    title="Restoran Öneri API",
    description="Kullanıcı yorumları ekleyebilen ve restoran önerisi yapan RAG sistemi.",
    version="1.0.0"
)


# --- 3. API Veri Modellerini Tanımlama (Pydantic) ---

class NewReviewModel(BaseModel):
    """Yeni bir yorum eklemek için gereken model."""
    restaurant_name: str = Field(..., description="Restoranın tam adı", example="Fıccın Restoran")
    review_text: str = Field(..., description="Kullanıcının yorum metni",example="Çerkez tavuğu harikaydı, porsiyonlar çok büyüktü.")
    location: Optional[str] = Field(None,example='41.032,28.979')



class QueryModel(BaseModel):
    """Sorgu yapmak için gereken model."""
    query: str = Field(..., description="Kullanıcının RAG sistemine sorusu",example="Beyoğlunda tavuk yiyebileceğim yerler.")


# --- 4. Sunucu Başlangıç Olayı (Startup Event) ---
# API sunucusu AYAĞA KALKARKEN SADECE BİR KEZ çalışır.
@app.on_event("startup")
def on_startup():
    global db_connection
    print("API sunucusu başlıyor...")

    db_file_path = os.path.join(DB_PERSIST_DIRECTORY, "chroma.sqlite3")
    if not os.path.exists(db_file_path):
        print("Veritabanı bulunamadı. CSV'den sıfırdan oluşturuluyor...")
        db_connection=build_database()
        print("Veritabanı oluşturuldu.")
    else:
        db_connection = db_def()
        print("Veritabanı zaten mevcut (db already exist).")

    # Adım 2: Modelleri "Isındırma"
    # Llama-3 ve Embedding modellerini sunucu başlarken belleğe yüklüyoruz.
    print("Modeller belleğe yükleniyor (ısındırma)...")
    try:
        # RAG pipeline'ını boş bir sorguyla tetikle
        rag_pipeline(db_connection,"test sorgusu")
        print("Tüm modeller başarıyla yüklendi. Sunucu hazır.")
    except Exception as e:
        print(f"Modeller yüklenirken bir hata oluştu: {e}")


# --- 5. API Endpoints (Servis Noktaları) ---

@app.get("/")
def read_root():
    """Ana dizin, API'nin çalıştığını doğrular."""
    return {
        "message": "Restoran Öneri API'sine hoş geldiniz! Test için /docs adresine gidin."
    }


@app.post("/add_review/", tags=["Yorum Yönetimi"])
async def api_add_new_review(review: NewReviewModel):
    """
    Veritabanına yeni bir kullanıcı yorumu ekler.
    Bu yorum ANINDA lokal veritabanına kaydedilir.
    """
    print(f"Yeni yorum alınıyor: {review.restaurant_name}")
    try:
        success = add_new_review(db_connection,
            restaurant_name=review.restaurant_name,
            review_text=review.review_text,
            location=review.location
        )
        if success:
            return {"status": "success", "message": "Yorum başarıyla eklendi."}
        else:
            return {"status": "error", "message": "Yorum eklenirken bir sunucu hatası oluştu."}
    except Exception as e:
        return {"status": "error", "message": f"Bir hata oluştu: {str(e)}"}


@app.post("/query/", tags=["RAG Sorgulama"],summary="RAG Pipeline'ını Çalıştır")
async def api_run_rag_query(request: QueryModel):
    """
    RAG sistemine bir sorgu gönderir ve yanıt alır.
    """
    print(f"Yeni sorgu alınıyor: {request.query}")
    try:
        # Ağır RAG işlemini çağır
        result = rag_pipeline(db_connection,request.query)
        return {"response": result['output_text']}
    except Exception as e:
        return {"status": "error", "message": f"Sorgu işlenirken bir hata oluştu: {str(e)}"}


# --- 6. Sunucuyu Çalıştırma ---
if __name__ == "__main__":
    # uvicorn sunucuyu 127.0.0.1 (localhost) adresinde 8000 portunda başlatır.
    uvicorn.run("server:app", host="127.0.0.1", port=8000)
