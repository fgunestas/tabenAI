from langchain_huggingface import HuggingFacePipeline

from components.retriever import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


def rest_prep():
    retriever = Retriever()

    query = "Beşiktaş’ta iyi lahmacun yapan yerler hangileri?"
    docs = retriever.get_relevant_documents(query)

    grouped = {}
    for doc in docs:
        rname = doc.metadata.get("restaurant")
        text = doc.page_content

        if text.startswith(f"[{rname}]"):
            text = text[len(rname) + 3:].strip()

        if rname not in grouped:
            grouped[rname] = []

        grouped[rname].append(text)

        structured = [
            {"name": rname, "reviews": text}
            for rname, text in grouped.items()
        ]

    return structured

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
        input_variables=["text", "question"],
        template="""
        Aşağıda tek bir restorana ait kullanıcı yorumları verilmiştir.
        Bu yorumları inceleyerek soruya uygun kısa bir özet çıkar.

        Kurallar:
        1. Restoranın adını başta belirt.
        2. Soruda geçen konuya (örneğin 'lahmacun') dair yorumlar varsa, sadece bu konudaki görüşleri özetle.
        3. Eğer sorulan konu hakkında yorum yoksa, "Bu restoran için bu konuda yorum bulunmuyor." de.
        4. Restoranın genel durumu (lezzet, fiyat/performans, servis) hakkında da 1-2 cümlelik genel özet yap.
        5. Kısa, net ve anlaşılır Türkçe cevap ver.

        --- Restoran ---
        

        --- Yorumlar ---
        {text}

        --- Soru ---
        {question}

        --- Cevap ---
        """
    )

    map_reduce_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt
    )
    resto_dict = rest_prep()

    docs = [
        Document(page_content="\n".join(rest["reviews"]))
        for rest in resto_dict
    ]

    query = "Beşiktaş’ta iyi lahmacun yapan yerler hangileri?"
    result = map_reduce_chain.invoke({"input_documents": docs, "question": query})
    print(result)











