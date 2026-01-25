"""
RAG Pipeline - Context Stuffing with LCEL
==========================================
Modern LangChain Expression Language (LCEL) kullanarak 
Context Stuffing tabanlı RAG pipeline.
"""

import os
from operator import itemgetter
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from components.retriever import build_context

# Environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# --- MODEL AYARLARI ---
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.1
RETRIEVAL_K = 50  # Çekilecek yorum sayısı

# --- PROMPT TEMPLATE ---
SYSTEM_PROMPT = """Sen uzman bir gurmesin. Görevin, kullanıcının restoran tercihine yardımcı olmaktır.

Sana verilen gruplanmış restoran yorumlarını dikkatlice analiz et ve kullanıcının isteğine en uygun restoranı öner.

**Kurallar:**
1. Cevabını mutlaka TÜRKÇE olarak yaz.
2. Önerini yaparken mutlaka yorumlardan doğrudan alıntı yap. Alıntıları tırnak içinde göster.
3. Neden o restoranı önerdiğini açıkla.
4. Eğer birden fazla uygun restoran varsa, karşılaştırmalı olarak değerlendir.
5. Yorumlarda yeterli bilgi yoksa, bunu açıkça belirt.

**Örnek alıntı formatı:**
Bir müşteri şöyle demiş: "Köfteleri muhteşemdi, kesinlikle tavsiye ederim."
"""

USER_PROMPT_TEMPLATE = """**Kullanıcının Sorusu:** {question}

**Restoran Yorumları:**
{context}

**Lütfen yukarıdaki yorumları analiz ederek kullanıcının sorusuna en uygun restoran önerisini yap:**"""

# --- GLOBAL CACHE ---
_cached_chain = None
_cached_llm = None


def _get_llm():
    """Gemini LLM instance'ını döndürür (cache'li)."""
    global _cached_llm
    
    if _cached_llm is None:
        print(f"Gemini API bağlantısı kuruluyor... (Model: {MODEL_NAME})")
        _cached_llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=TEMPERATURE,
            convert_system_message_to_human=True
        )
        print("Gemini bağlantısı başarılı.")
    
    return _cached_llm


def _create_prompt():
    """ChatPromptTemplate oluşturur."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE)
    ])
    return prompt


def _build_chain(db):
    """
    LCEL kullanarak RAG chain oluşturur.
    
    Akış: Soru -> Retrieval -> Formatlama -> Prompt -> Gemini -> Output
    """
    llm = _get_llm()
    prompt = _create_prompt()
    
    # Retrieval fonksiyonunu RunnableLambda ile wrap et
    def retrieve_and_format(inputs):
        query = inputs["question"]
        context = build_context(db, query, k=RETRIEVAL_K)
        return context
    
    # LCEL Chain
    chain = (
        {
            "question": itemgetter("question"),
            "context": RunnableLambda(retrieve_and_format)
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def initialize_pipeline(db):
    """
    Pipeline'ı başlatır ve cache'e alır.
    Sunucu başlangıcında çağrılabilir.
    """
    global _cached_chain
    
    if _cached_chain is None:
        print("RAG Pipeline başlatılıyor...")
        _cached_chain = _build_chain(db)
        print("RAG Pipeline başarıyla hazırlandı.")
    
    return _cached_chain


def rag_pipeline(db, query: str) -> dict:
    """
    RAG Pipeline'ı çalıştırır.
    
    Args:
        db: ChromaDB vector store instance
        query: Kullanıcının sorusu
    
    Returns:
        dict: {"output_text": "..."} formatında sonuç
    """
    global _cached_chain
    
    try:
        # Chain'i oluştur veya cache'den al
        if _cached_chain is None:
            _cached_chain = _build_chain(db)
        
        print(f"Sorgu işleniyor: '{query}'")
        
        # Chain'i çalıştır
        result = _cached_chain.invoke({"question": query})
        
        print("Sorgu başarıyla işlendi.")
        return {"output_text": result}
    
    except Exception as e:
        error_msg = f"RAG Pipeline hatası: {str(e)}"
        print(f"HATA: {error_msg}")
        return {"output_text": error_msg}


def rag_stream(db, query: str):
    """
    RAG Pipeline'ı streaming modda çalıştırır.
    
    Args:
        db: ChromaDB vector store instance
        query: Kullanıcının sorusu
    
    Yields:
        str: Token'lar birer birer
    """
    global _cached_chain
    
    try:
        if _cached_chain is None:
            _cached_chain = _build_chain(db)
        
        print(f"Streaming sorgu: '{query}'")
        
        for chunk in _cached_chain.stream({"question": query}):
            yield chunk
    
    except Exception as e:
        yield f"Hata: {str(e)}"