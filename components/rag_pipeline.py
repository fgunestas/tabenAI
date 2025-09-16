from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from components.retriever import Retriever
from transformers import pipeline

def rag_pipeline(query):

    llm_model="mistralai/Mistral-7B-Instruct-v0.2"
    generate=pipeline("text-generation", model=llm_model, temperature=0.6,use_auth_token=True)

    llm=HuggingFacePipeline(pipeline=generate)


    qa=RetrievalQA.from_chain_type(llm=llm, retriever=Retriever,chain_type="stuff")
    query = "Beşiktaş’ta iyi lahmacun yapan yerler hangileri?"
    result = qa.run(query)
    print(result)
