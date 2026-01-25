from typing import List, Dict
from langchain_core.documents import Document
from collections import defaultdict


def get_retriever(db, k: int = 50):
    """
    VectorDB için retriever objesi döndürür.
    MMR (Maximal Marginal Relevance) kullanarak çeşitlilik sağlar.
    """
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            'k': k,
            'fetch_k': k * 3  # MMR için daha fazla aday getir
        }
    )
    return retriever


def retrieve_relevant_reviews(db, query: str, k: int = 50) -> List[Document]:
    """
    Kullanıcı sorusuna göre VectorDB'den en alakalı k yorumu çeker.
    
    Args:
        db: ChromaDB vector store instance
        query: Kullanıcının sorusu
        k: Çekilecek maksimum yorum sayısı (default: 50)
    
    Returns:
        List[Document]: Alakalı dokümanların listesi
    """
    retriever = get_retriever(db, k=k)
    docs = retriever.invoke(query)
    return docs


def group_reviews_by_restaurant(docs: List[Document]) -> str:
    """
    Çekilen dokümanları restoran ismine göre gruplar ve formatlanmış string döndürür.
    
    Args:
        docs: VectorDB'den çekilen Document listesi
    
    Returns:
        str: Restoranlara göre gruplandırılmış yorumlar
        
    Örnek çıktı:
        ### Restoran: Baba Döner
        - Yorum: Yemekler çok lezzetliydi.
        - Yorum: Servis hızlıydı.
        
        ### Restoran: Köfteci Yusuf
        - Yorum: Köfteler harika.
    """
    # Restoranlara göre grupla
    restaurant_reviews: Dict[str, List[str]] = defaultdict(list)
    
    for doc in docs:
        # Metadata'dan restoran adını al
        restaurant_name = doc.metadata.get("restaurant", "Bilinmeyen Restoran")
        
        # page_content'ten yorum metnini çıkar
        # Format: "[Restoran adı: X] Yorum metni"
        content = doc.page_content
        
        # Prefix'i kaldırarak sadece yorum metnini al
        if content.startswith("[Restoran adı:"):
            # "]" karakterinden sonrasını al
            bracket_end = content.find("]")
            if bracket_end != -1:
                review_text = content[bracket_end + 1:].strip()
            else:
                review_text = content
        else:
            review_text = content
        
        if review_text:  # Boş yorumları ekleme
            restaurant_reviews[restaurant_name].append(review_text)
    
    # Formatlanmış string oluştur
    output_parts = []
    
    for restaurant_name, reviews in restaurant_reviews.items():
        section = f"### Restoran: {restaurant_name}"
        for review in reviews:
            section += f"\n- Yorum: {review}"
        output_parts.append(section)
    
    return "\n\n".join(output_parts)


def build_context(db, query: str, k: int = 50) -> str:
    """
    Kullanıcı sorusuna göre VectorDB'den yorumları çeker ve 
    context stuffing için hazır bir string döndürür.
    
    Bu fonksiyon retrieve_relevant_reviews ve group_reviews_by_restaurant
    fonksiyonlarını birleştirir.
    
    Args:
        db: ChromaDB vector store instance
        query: Kullanıcının sorusu
        k: Çekilecek maksimum yorum sayısı (default: 50)
    
    Returns:
        str: LLM'e gönderilmeye hazır formatlanmış context string
    """
    docs = retrieve_relevant_reviews(db, query, k=k)
    context = group_reviews_by_restaurant(docs)
    return context


# Geriye dönük uyumluluk için eski fonksiyon adını koru
def Retriever(db):
    """
    [DEPRECATED] Eski API uyumluluğu için korunmuştur.
    Yeni kodlarda get_retriever() kullanın.
    """
    return get_retriever(db, k=15)
