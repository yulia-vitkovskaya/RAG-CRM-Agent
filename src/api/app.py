from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.generation.generator import LocalLLMGenerator, TinyLlamaLLM

app = FastAPI(title="RAG CRM Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

qa_chain = None

@app.on_event("startup")
async def load_components():
    global qa_chain

    embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectordb_path = "data/vectorstore/index"
    vectorstore = FAISS.load_local(vectordb_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Ты — ассистент по документации CRM. Отвечай только по приведённому контексту, кратко и профессионально.

Требования к ответу:
- Используй только информацию из контекста.
- Не выдумывай ничего.
- Если ответа в контексте нет — скажи: "Контекста недостаточно для ответа на вопрос."

Контекст:
{context}

Вопрос: {question}
Ответ:"""
    )

    llm = TinyLlamaLLM(generator=LocalLLMGenerator())
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(payload: QueryRequest):
    global qa_chain

    if not qa_chain:
        return QueryResponse(answer="Система не готова", sources=[])

    result = qa_chain.invoke({"query": payload.question})
    answer = result["result"]
    sources = list({doc.metadata.get("source", "unknown") for doc in result["source_documents"]})

    return QueryResponse(answer=answer, sources=sources)
