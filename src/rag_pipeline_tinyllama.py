from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.generation.generator import LocalLLMGenerator, TinyLlamaLLM


SYSTEM_PROMPT = """Ты — ассистент по документации CRM. Отвечай только по приведённому контексту, кратко и профессионально.

Требования к ответу:
- Используй только информацию из контекста.
- Не выдумывай ничего.
- Если ответа в контексте нет — скажи: "Контекста недостаточно для ответа на вопрос."

Контекст:
{context}

Вопрос: {question}
Ответ:"""


def run_tinyllama_rag(question: str, vectordb_path: str = "data/vectorstore/index") -> dict:
    """
    Запускает RAG-пайплайн с TinyLlama и возвращает ответ и источники.

    :param question: Вопрос пользователя.
    :param vectordb_path: Путь к сохранённому FAISS-индексу.
    :return: Словарь с полями "result" (ответ) и "source_documents" (список документов).
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.load_local(vectordb_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT
    )

    generator = LocalLLMGenerator("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    llm = TinyLlamaLLM(generator=generator)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    result = qa_chain({"query": question})
    return result
