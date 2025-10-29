import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def add_new_review(db_store,restaurant_name: str, review_text: str, location: str = None):


    """
    Veritabanına yeni bir kullanıcı yorumu ekler.
    Eğer restoran varsa günceller, yoksa yeni bir belge oluşturur.
    """
    print(f"'{restaurant_name}' için yeni yorum ekleniyor...")

    try:
        # Adım 1: Restoranın veritabanında zaten var olup olmadığını KONTROL ET
        # Not: Bu, 'restaurant_name'in metadata'da olduğunu varsayar.
        existing_doc = db_store.get(
            where={"restaurant_name": restaurant_name},
            include=["metadatas", "documents"]  # <-- "ids" kelimesini buradan kaldırın
        )

        # Senaryo A: Restoran BULUNAMADI (Listesi boş geldi)
        if not existing_doc or not existing_doc.get('ids'):
            print(f"'{restaurant_name}' bulunamadı. Yeni bir belge oluşturuluyor.")

            # 1. Yeni belgenin içeriğini formatla
            page_content = f"Restoran Adı: {restaurant_name}\n\nYorumlar:\n: {review_text}"

            # 2. Metadata'yı oluştur
            metadata = {"restaurant_name": restaurant_name}
            if location:
                metadata["location"] = location

            # 3. LangChain Belge nesnesini oluştur
            new_document = Document(page_content=page_content, metadata=metadata)

            # 4. Veritabanına EKLE
            # (ID vermediğimiz için Chroma yeni bir ID üretecek)
            db_store.add_documents([new_document])
            print(f"Başarılı: '{restaurant_name}' ve ilk yorumu eklendi.")

        # Senaryo B: Restoran BULUNDU (Güncelleme yapacağız)
        else:
            print(f"'{restaurant_name}' bulundu. Mevcut belge güncelleniyor...")

            # 1. Eski belgenin bilgilerini al
            doc_id = existing_doc['ids'][0]
            old_page_content = existing_doc['documents'][0]
            old_metadata = existing_doc['metadatas'][0]

            # 2. Yeni yorumu eski içeriğin SONUNA EKLE
            new_review_line = f"\n: {review_text}"
            new_page_content = old_page_content + new_review_line

            # 3. Güncellenmiş LangChain Belge nesnesini oluştur
            updated_document = Document(page_content=new_page_content, metadata=old_metadata)

            # 4. Veritabanını GÜNCELLE
            # Chroma'da güncellemenin modern yolu, MEVCUT ID'yi
            # belirterek .add_documents() çağırmaktır.
            # Bu, o ID'ye sahip belgeyi ezer (overwrite/update).
            db_store.add_documents(
                documents=[updated_document],
                ids=[doc_id]  # <-- Anahtar burası: Hangi belgenin güncelleneceğini söyler
            )
            print(f"Başarılı: '{restaurant_name}' için yeni yorum eklendi.")

    except Exception as e:
        print(f"HATA: Yorum eklenirken bir sorun oluştu: {e}")