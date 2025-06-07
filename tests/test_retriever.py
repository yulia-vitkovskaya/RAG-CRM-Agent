from src.retrieval.retriever import DocumentRetriever

retriever = DocumentRetriever()
results = retriever.search("Как создать клиента в CRM?", top_k=2)

for res in results:
    print(f"[{res['source']} стр.{res['page']}]\n{res['chunk']}\n---")

