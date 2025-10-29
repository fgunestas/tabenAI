def Retriever(db):
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            'k': 15,      # Nihai olarak 10 belge istiyoruz
            'fetch_k': 40 # MMR'ın aralarından seçim yapması için 50 aday belge getir
        }
    )


    return retriever
