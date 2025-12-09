import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

from components.retriever import Retriever

# --- 1. PROMPT'LAR ---

reduce_prompt = PromptTemplate(
    input_variables=["text", "question"],
    template="""
Sen bir baş restoran öneri uzmanısın. Görevin, farklı restoranlar için hazırlanmış analiz özetlerini kullanarak kullanıcının "{question}"suna kapsamlı ve nihai bir yanıt oluşturmaktır.
**Orijinal Soru:** {question}
**Restoran Analiz Özetleri:** {text}
**Talimatlar:**
1.  Kullanıcının orijinal sorusunu ({question}) anla. Soru "konum" (örn: Beyoğlu) ve "istek" (örn: tavuk) içeriyorsa dikkat et.
2.  Özetleri incele. SADECE sorudaki "konum" bilgisine uyan restoranları nihai önerine dahil et. (Örn: Soru "Beyoğlu" ise, "Üsküdar" özetini DAHİL ETME.)
3.  Cevabını mutlaka **TÜRKÇE** olarak yaz.
**Nihai Öneri ve Karşılaştırma (TÜRKÇE):**
"""
)

map_prompt = PromptTemplate(

    input_variables=["text", "question"],
    template="""Sen bir restoran yorumu analiz asistanısın. Görevin, verilen yorumları "{question}" sorusu açısından analiz etmektir.
**Kullanıcı Sorusu:** {question}
**Bu Restorana Ait Yorumlar:** {text}
**Talimatlar:**
1.  Yorumlarda soruyla (örn: tavuk, tavuk pirzola) ilgili spesifik bir bilgi var mı?
2.  **BİLGİ VARSA:** Sadece bu bilgiyi 1-2 cümle ile özetle. (Örn: "Yorumlarda 'tavuk kanat' ve 'beğendili tavuk' geçtiği, lezzetli olduğu belirtilmiş.")
3.  **BİLGİ YOKSA:** Başka HİÇBİR ŞEY YAZMA. Sadece şu cümleyi yaz: "Yorumlarda bu soruya yönelik spesifik bir bilgi bulunmamaktadır."
4.  **Kural:** ASLA kendini tekrar etme. Cevabın toplam 100 kelimeyi geçmesin.
**Soruya Yönelik Kısa Özet:**
"""
)

# --- 2. GLOBAL ÖNBELLEK (CACHE) ---

_cached_map_reduce_chain = None


# --- 3. AĞIR MODELLERİ YÜKLEYEN FONKSİYON ---
def _initialize_pipeline():
    """
    Gemini modelini ve zinciri hazırlar. (VRAM kullanmaz, çok hızlıdır)
    """
    global _cached_map_reduce_chain

    if _cached_map_reduce_chain is not None:
        return

    print("Gemini API bağlantısı kuruluyor...")

    # --- GEMINI MODELİ ---

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        convert_system_message_to_human=True
    )

    print("MapReduce Zinciri oluşturuluyor...")

    # 1. Map Zinciri
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # 2. Reduce Zinciri
    reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain,
        document_variable_name="text"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=30000  # Gemini'nin context window'u çok geniştir (1M token), sınır sorunu yok.
    )

    # 3. Ana Zincir
    _cached_map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="text",
        verbose=True
    )

    print("Gemini RAG Pipeline başarıyla hazırlandı.")



# --- 4. VERİ HAZIRLAMA FONKSİYONU ---

def rest_prep(db, query: str) -> List[Document]:
    """
    Veritabanından belgeleri alır ve restoran adına göre gruplar.
    """

    retriever = Retriever(db)
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return []

    # 1. Adım: Yorumları restorana göre grupla
    grouped = {}
    for doc in docs:
        rname = doc.metadata.get("restaurant")
        if not rname:
            continue


        text = doc.page_content
        if text.startswith(f"[{rname}]"):
            text = text[len(rname) + 2:].strip()

        if rname not in grouped:
            grouped[rname] = []
        grouped[rname].append(text)

    # 2. Adım: Grupları tek 'Document' objelerine dönüştür
    sonuc_belgeleri = []
    for rname, reviews in grouped.items():
        birlesik_yorumlar = "\n: ".join(reviews)  # Okunabilirlik için ": " ekledim
        final_content = f"Restoran Adı: {rname}\n\nYorumlar:\n: {birlesik_yorumlar}"

        new_doc = Document(
            page_content=final_content,
            metadata={"restaurant_name": rname}
        )
        sonuc_belgeleri.append(new_doc)

    return sonuc_belgeleri


# --- 5. ANA RAG FONKSİYONU ---
# server.py'nin çağıracağı ASIL fonksiyon budur.

def rag_pipeline(db, query: str):
    """
    Ağır modelleri/zinciri (gerekirse) bir kez yükler,
    verilen sorgu için RAG pipeline'ını çalıştırır.
    """

    # Adım 1: Her şeyin yüklü olduğundan emin ol
    # Bu, 'on_startup'taki "test sorgusu" sırasında modelleri yükleyecek.

    try:
        _initialize_pipeline()
    except Exception as e:
        print(f"MODEL YÜKLENİRKEN KRİTİK HATA: {e}")
        return {"output_text": f"Model yüklenirken bir hata oluştu: {e}"}


    print(f"'{query}' için belgeler hazırlanıyor...")
    resto_dict = rest_prep(db, query)


    resto_dict = sorted(resto_dict, key=lambda doc: len(doc.page_content), reverse=True)[:5]

    if not resto_dict:
        print("Sorguya uygun belge bulunamadı.")
        return {"output_text": "Yorumlarda bu soruya uygun spesifik bir bilgi bulunamadı."}

    # Adım 3: Zinciri Çalıştır
    # (Artık ağır modelleri yüklemez, sadece önbellekteki zinciri kullanır)
    print("MapReduce zinciri çalıştırılıyor...")
    result = _cached_map_reduce_chain.invoke({
        "input_documents": resto_dict,
        "question": query
    })


    return result