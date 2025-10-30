import uvicorn  # API sunucumuzu çalıştırmak için
from fastapi import FastAPI
from pydantic import BaseModel, Field  # Gelen verileri doğrulamak için
from typing import Optional
import os

# --- 1. Kendi Modüllerimizi Import Ediyoruz ---
#    'vector_store.py' dosyamızdan gerekli fonksiyonları alıyoruz
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
# Bu, FastAPI'ye hangi verinin geleceğini söyler ve otomatik doğrular.

class NewReviewModel(BaseModel):
    """Yeni bir yorum eklemek için gereken model."""
    restaurant_name: str = Field(..., description="Restoranın tam adı")
    review_text: str = Field(..., description="Kullanıcının yorum metni")
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class QueryModel(BaseModel):
    """Sorgu yapmak için gereken model."""
    query: str = Field(..., description="Kullanıcının RAG sistemine sorusu")


# --- 4. Sunucu Başlangıç Olayı (Startup Event) ---
# API sunucusu AYAĞA KALKARKEN SADECE BİR KEZ çalışır.
@app.on_event("startup")
def on_startup():
    global db_connection
    print("API sunucusu başlıyor...")

    # Adım 1: main.py'deki veritabanı kontrolünü yap
    db_file_path = os.path.join(DB_PERSIST_DIRECTORY, "chroma.sqlite3")
    if not os.path.exists(db_file_path):
        print("Veritabanı bulunamadı. CSV'den sıfırdan oluşturuluyor...")
        build_database()
        print("Veritabanı oluşturuldu.")
    else:
        db_connection = db_def()
        print("Veritabanı zaten mevcut (db already exist).")

    # Adım 2: Modelleri "Isındırma" (Warm-up)
    # İlk kullanıcının 1 dakika beklemesini önlemek için,
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


@app.post("/add_review/")
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
            latitude=review.latitude,
            longitude=review.longitude
        )
        if success:
            return {"status": "success", "message": "Yorum başarıyla eklendi."}
        else:
            return {"status": "error", "message": "Yorum eklenirken bir sunucu hatası oluştu."}
    except Exception as e:
        return {"status": "error", "message": f"Bir hata oluştu: {str(e)}"}


@app.post("/query/")
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
    # Bu script'i 'python server.py' olarak çalıştırdığınızda,
    # uvicorn sunucuyu 127.0.0.1 (localhost) adresinde 8000 portunda başlatır.
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
    # reload=True: Kodu her kaydettiğinizde sunucuyu otomatik yeniden başlatır.