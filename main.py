"""
TabenAI - Restoran Öneri Sistemi
Test ve Ana Çalıştırma Dosyası
"""

import subprocess
from components.rag_pipeline import rag_pipeline
from components.vector_store import main as build_database, db_def
from components.add_review import add_new_review
from components.retriever import retrieve_relevant_reviews
import os


def test_rag_system():
    """
    RAG sistemini test eder ve detaylı log çıktısı verir.
    """
    print("=" * 80)
    print("RAG SİSTEM TESTİ BAŞLIYOR")
    print("=" * 80)
    
    # 1. Veritabanını yükle
    print("\n[1/4] Veritabanı kontrol ediliyor...")
    db = db_def()
    print("✓ Veritabanı hazır.")
    
    # 2. Test sorusu
    test_query = "Beşiktaş'ta sessiz sakin tavuğu güzel bir yer var mı?"
    print(f"\n[2/4] Test Sorusu: '{test_query}'")
    
    # 3. Retrieval testi (ham veri)
    print("\n[3/4] Retrieval işlemi yapılıyor...")
    try:
        retrieved_docs = retrieve_relevant_reviews(db, test_query, k=50)
        print(f"✓ Retrieve edilen yorum sayısı: {len(retrieved_docs)}")
        
        # İlk 3 yorumu göster (debug için)
        print("\n📋 İlk 3 Yorum Örneği:")
        for i, doc in enumerate(retrieved_docs[:3], 1):
            restaurant = doc.metadata.get('restaurant', 'Bilinmiyor')
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  {i}. [{restaurant}] {content_preview}")
    
    except Exception as e:
        print(f"✗ Retrieval hatası: {e}")
        return
    
    # 4. RAG Pipeline testi (Gemini'ye gönder)
    print("\n[4/4] RAG Pipeline çalıştırılıyor...")
    print("-" * 80)
    
    try:
        result = rag_pipeline(db, test_query)
        
        print("\n🤖 GEMİNİ CEVABI:")
        print("=" * 80)
        print(result.get("output_text", "Sonuç alınamadı."))
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ RAG Pipeline hatası: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ Test tamamlandı!")


def main():
    """
    Ana fonksiyon - Veritabanı oluşturma ve test
    """
    print("\n🚀 TabenAI Başlatılıyor...\n")
    
    # Veritabanı kontrolü
    if os.path.exists("./chroma_store/chroma.sqlite3"):
        print("✓ Veritabanı mevcut (chroma_store bulundu)")
        print("  Not: Yeniden oluşturmak için './chroma_store' klasörünü silin.\n")
    else:
        print("⚠ Veritabanı bulunamadı. Oluşturuluyor...")
        print("  Bu işlem birkaç dakika sürebilir...\n")
        build_database()
        print("\n✓ Veritabanı başarıyla oluşturuldu!\n")
    
    # RAG sistemini test et
    test_rag_system()


if __name__ == "__main__":
    main()