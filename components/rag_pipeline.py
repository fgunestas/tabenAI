from langchain_huggingface import HuggingFacePipeline

from components.retriever import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from typing import List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


def rest_prep(db,query: str) -> List[Document]:
    retriever = Retriever(db)

    docs = retriever.get_relevant_documents(query)

    # 1. Adım: Yorumları restorana göre doğru şekilde grupla
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

    # 2. Adım: Gruplanan yorumları string yerine Document objelerine dönüştür
    sonuc_belgeleri = []
    for rname, reviews in grouped.items():
        # Yorumları tek bir metin bloğunda birleştir
        birlesik_yorumlar = "\n".join(reviews)

        # Restoran adını ve yorumları içeren bir metin oluştur.
        # Bu, prompt'a direkt olarak gidecek içerik olacak.
        final_content = f"Restoran Adı: {rname}\n\nYorumlar:\n{birlesik_yorumlar}"

        # String yerine Document objesi oluştur ve listeye ekle
        new_doc = Document(
            page_content=final_content,
            metadata={"restaurant_name": rname}  # Metadata'yı da ekleyebiliriz
        )
        sonuc_belgeleri.append(new_doc)
    return sonuc_belgeleri

def rag_pipeline(db,query):

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )



    #llm_model="mistralai/Mistral-7B-Instruct-v0.2"
    #llm_model="EleutherAI/gpt-neo-1.3B"
    llm_model="meta-llama/Meta-Llama-3-8B-Instruct"

    tok = AutoTokenizer.from_pretrained(llm_model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id


    model = AutoModelForCausalLM.from_pretrained(
        llm_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16
    )



    generate=pipeline("text-generation",
                      model=model,
                      temperature=0.2,
                      tokenizer=tok,
                      do_sample=True,
                      max_new_tokens=512,
                      clean_up_tokenization_spaces=True,
                      return_full_text=False)

    llm=HuggingFacePipeline(pipeline=generate)

    reduce_prompt = PromptTemplate(
        input_variables=["text", "question"],
        template="""
Sen bir baş restoran öneri uzmanısın. Görevin, farklı restoranlar için hazırlanmış analiz özetlerini kullanarak kullanıcının "{question}"suna kapsamlı ve nihai bir yanıt oluşturmaktır.

**Talimatlar:**
1.  Kullanıcının orijinal "{question}"sunu anla.
2.  Aşağıdaki özetler listesini incele. Her bir özet, farklı bir restoranın "{question}"ya ne kadar uygun olduğunu belirtir.
3.  Bu özetleri sentezleyerek kullanıcıya en iyi öneriyi sun.
4.  Yanıtın, doğrudan "{question}"yu hedef almalı ve sadece sağlanan özetlerdeki bilgilere dayanmalıdır.
5.  Hangi restoranın (veya restoranların) soruya en uygun olduğunu açıkça belirt. Mümkünse, seçenekleri kısaca karşılaştır veya hangi restoranın hangi konuda (örn: "X restoranı sakinlik, Y restoranı lezzet konusunda") öne çıktığını belirt.
6.  Eğer özetlerin çoğu yetersiz bilgi içeriyorsa, net bir öneri yapmanın zor olduğunu ifade et.

**Orijinal Soru:**
{question}

**Restoran Analiz Özetleri:**
{text}

**Nihai Öneri ve Karşılaştırma:**
"""
    )
    map_prompt = PromptTemplate(
        input_variables=["text", "question", "restaurant_name"],
        template="""Sen bir restoran yorumu analiz asistanısın. Görevin, verilen yorumları {question} açısından analiz etmektir.

**Talimatlar:**
1.  {question}'yu ve yorumlar'ı oku.
2.  Yorumlarda {question} ile ilgili (olumlu veya olumsuz) spesifik bir bilgi (örn: tavuk, tavuk pirzola, tavuk şiş, çerkez tavuğu vb.) var mı diye analiz et.

**Çıktı Kuralları (ÇOK ÖNEMLİ):**
* **EĞER BİLGİ VARSA:** Bilgiyi 1-2 cümleyle özetle. Yorumlardan 1-2 kelimelik kısa bir alıntı ekle. **Özetin 3 cümleyi GEÇMEMELİDİR.**
* **EĞER BİLGİ YOKSA:** Başka HİÇBİR ŞEY YAZMA. Sadece şu tek cümleyi yaz: "Yorumlarda bu soruya yönelik spesifik bir bilgi bulunmamaktadır."
* Varsayım yapma. Yorumda olmayan bilgiyi ekleme.
* Konu dışı yorumları (meze, balık, servis hızı vb.) özetine DAHİL ETME.

**Kullanıcı Sorusu:**
{question}

**Bu Restorana Ait Yorumlar:**
{text}

**Soruya Yönelik Özet:**
"""
    )
    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=True,
        token_max=8192
    )

    from langchain.chains import StuffDocumentsChain

    # Map'ten gelen özetleri birleştirecek olan zincir


    resto_dict = rest_prep(db,"Beyoğlunda tavuk yiyebileceğim yerler.")
    resto_dict = sorted(resto_dict, key=lambda doc: len(doc.page_content), reverse=True)[:5]


    query = "Beyoğlunda tavuk yiyebileceğim yerler."
    result = map_reduce_chain.invoke({"input_documents": resto_dict, "question": query})
    print(result)










