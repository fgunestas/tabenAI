from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from components.retriever import Retriever
from transformers import pipeline

def rag_pipeline(query):

    #llm_model="mistralai/Mistral-7B-Instruct-v0.2"
    llm_model="EleutherAI/gpt-neo-1.3B"
    generate=pipeline("text-generation",
                      model=llm_model,
                      temperature=0.6,
                      do_sample=True,
                      max_new_tokens=256,
                      device=0)

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
