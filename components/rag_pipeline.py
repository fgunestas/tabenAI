from langchain_huggingface import HuggingFacePipeline

from components.retriever import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from typing import List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


def rest_prep(query: str) -> List[Document]:
    retriever = Retriever()

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

def rag_pipeline(query):

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )



    llm_model="mistralai/Mistral-7B-Instruct-v0.2"
    #llm_model="EleutherAI/gpt-neo-1.3B"

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
                      temperature=0.6,
                      tokenizer=tok,
                      do_sample=True,
                      max_new_tokens=128,
                      clean_up_tokenization_spaces=True)

    llm=HuggingFacePipeline(pipeline=generate)
    reduce_prompt = PromptTemplate(
        input_variables=["text", "question"],
        template="""
    Aşağıda farklı restoranların özetleri var.
    Bunları kullanarak soruya uygun genel bir cevap üret.

    --- Restoran Özetleri ---
    {text}

    --- Soru ---
    {question}

    --- Genel Cevap ---
    """
    )
    map_prompt = PromptTemplate(
        input_variables=["text", "question", "restaurant_name"],
        template="""
    Aşağıda tek bir restorana ait kullanıcı yorumları verilmiştir.
    Bu yorumları inceleyerek soruya uygun kısa bir özet çıkar.

    Kurallar:
    1. Restoranın adını başta belirt.
    2. Soruda geçen konuya (örneğin 'lahmacun') dair yorumlar varsa, sadece bu konudaki görüşleri özetle.
    3. Eğer sorulan konu hakkında yorum yoksa, "Bu restoran için bu konuda yorum bulunmuyor." de.
    4. Restoranın genel durumu (lezzet, fiyat/performans, servis) hakkında da 1-2 cümlelik genel özet yap.
    5. Kısa, net ve anlaşılır Türkçe cevap ver.


    --- Yorumlar ---
    {text}

    --- Soru ---
    {question}

    --- Cevap ---
    """
    )
    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=True
    )

    from langchain.chains import StuffDocumentsChain

    # Map'ten gelen özetleri birleştirecek olan zincir


    resto_dict = rest_prep("Beşiktaş’ta iyi lahmacun yapan yerler hangileri?")
    resto_dict = sorted(resto_dict, key=lambda doc: len(doc.page_content), reverse=True)[:5]


    query = "Beşiktaş’ta iyi lahmacun yapan yerler hangileri?"
    result = map_reduce_chain.invoke({"input_documents": resto_dict, "question": query})
    print(result)











