import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


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
    Tüm RAG pipeline'ını (Modeller + Zincir) SADECE BİR KEZ yükler.
    """
    global _cached_map_reduce_chain

    # Eğer zaten yüklüyse, tekrar yükleme
    if _cached_map_reduce_chain is not None:
        print("RAG Pipeline zaten yüklü, önbellekten kullanılıyor.")
        return

    print("RAG Pipeline (Llama-3) İLK KEZ yükleniyor...")
    print("Bu işlem GPU VRAM durumuna göre birkaç dakika sürebilir.")


    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    llm_model = "meta-llama/Meta-Llama-3-8B-Instruct"

    tok = AutoTokenizer.from_pretrained(llm_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,

    )


    print("MAP (kısa) pipeline'ı oluşturuluyor...")
    generate_map = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        temperature=0.1,
        do_sample=True,
        max_new_tokens=80,
        clean_up_tokenization_spaces=True,
        return_full_text=False
    )
    llm_map = HuggingFacePipeline(pipeline=generate_map)

    print("REDUCE (uzun) pipeline'ı oluşturuluyor...")
    generate_reduce = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        temperature=0.2,
        do_sample=True,
        max_new_tokens=1024,
        clean_up_tokenization_spaces=True,
        return_full_text=False
    )
    llm_reduce = HuggingFacePipeline(pipeline=generate_reduce)


    print("MapReduce Zinciri (MANUEL) oluşturuluyor...")

    # 1. Map Zinciri (llm_map'i kullanır)
    map_chain = LLMChain(llm=llm_map, prompt=map_prompt)

    # 2. Reduce Zinciri (llm_reduce'u kullanır)
    reduce_llm_chain = LLMChain(llm=llm_reduce, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain,
        document_variable_name="text"  # reduce_prompt'taki {text}
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=8192  # Llama-3'ün gerçek limiti
    )

    # 3. Ana Zinciri oluştur ve global önbelleğe kaydet
    _cached_map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="text",  # map_prompt'taki {text}
        verbose=True
    )

    print("RAG Pipeline (Llama-3) başarıyla yüklendi ve önbelleğe alındı.")


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