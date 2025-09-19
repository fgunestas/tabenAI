from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from components.retriever import Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

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
    retriever=Retriever()

    prompt_template = """
    Aşağıda kullanıcı yorumları verilmiştir. Yorumlar farklı restoranlara aittir. 
    Her restoran için özet çıkarırken şu kurallara uy:
    
    1. Restoranı adıyla belirt.
    2. Soruda geçen konuya (örneğin 'lahmacun') dair yorumlar varsa onları özetle.
    3. Eğer sorulan konu hakkında yorum yoksa, "Bu bölgede bu tarz yorum bulunmuyor." de.
    4. Genel restoran hakkında da 1-2 cümlelik özet yap (ör: lezzet, fiyat/performans, servis).
    5. Kısa, net ve anlaşılır Türkçe cevap ver. Liste formatında sun.
    
    --- Kullanıcı Yorumları ---
    {context}
    
    --- Soru ---
    {question}
    
    --- Cevap ---
    """
    PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa=RetrievalQA.from_chain_type(llm=llm,
                                   retriever=retriever,
                                   chain_type="stuff",
                                   chain_type_kwargs={"prompt": PROMPT})
    query = "Beşiktaş’ta iyi lahmacun yapan yerler hangileri?"
    result = qa.invoke(query)
    print(result)
