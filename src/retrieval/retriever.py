from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class DocumentRetriever:
    def __init__(self, vectordb_path="data/vectorstore/index"):
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = FAISS.load_local(
            vectordb_path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def search(self, query: str, top_k: int = 3):
        docs = self.retriever.get_relevant_documents(query)
        results = []
        for doc in docs[:top_k]:
            results.append({
                "chunk": doc.page_content,
                "source": doc.metadata.get("source", "N/A"),
                "page": doc.metadata.get("page", "N/A")
            })
        return results
