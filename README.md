# 🤖 RAG-CRM-Agent

Интеллектуальный ассистент, который отвечает на вопросы по документации CRM с помощью локальной LLM TinyLlama и RAG-подхода.

## Запуск

### Зависимости
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

### 2. Векторная база (FAISS)
python scripts/build_vectorstore.py

### 3. Прогон тестов
python -m tests.test_retriever
python -m tests.test_generator

### 4. Запуск API
```bash
uvicorn src.api.app:app --reload
```

Swagger будет доступен по адресу: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Docker

docker build -t rag-crm-agent .
docker run -p 8000:8000 rag-crm-agent
[http://localhost:8000/docs](http://localhost:8000/docs)

### Структура проекта
```text
├── data/                  # Документы и векторная база
│   ├── source_docs/
│   └── vectorstore/
├── scripts/              # Сборка индекса
├── src/
│   ├── api/              # FastAPI
│   ├── generation/       # Модель + генерация
│   ├── indexing/         # Индексация документации
│   ├── retrieval/        # Поиск в базе
├── tests/                # Юнит-тесты
├── Dockerfile
├── start.sh              # Uvicorn-стартер
├── requirements.txt
└── README.md
```

