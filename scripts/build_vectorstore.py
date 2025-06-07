from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import os

def load_documents(source_dir: str):
    docs = []
    for file in os.listdir(source_dir):
        path = os.path.join(source_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".md") or file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    return docs

def build_vectorstore():
    source_dir = "data/source_docs"
    persist_dir = "data/vectorstore/index"

    print("Загружаем документы")
    documents = load_documents(source_dir)

    print(f"Разбиение на чанки")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    print("Генерация эмбеддингов и построение индекса")
    embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.from_documents(texts, embeddings)

    db.save_local(persist_dir)
    print(f"Индекс сохранён в: {persist_dir}")

if __name__ == "__main__":
    build_vectorstore()
