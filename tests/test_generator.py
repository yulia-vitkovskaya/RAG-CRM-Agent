from src.rag_pipeline_tinyllama import run_tinyllama_rag

def test_rag_pipeline():
    question = "Что такое CRM?"
    result = run_tinyllama_rag(question)

    assert isinstance(result, dict)
    assert "result" in result
    assert "source_documents" in result
    assert isinstance(result["result"], str)
    assert len(result["result"]) > 5

    print("Ответ:", result["result"])

if __name__ == "__main__":
    test_rag_pipeline()
    print("Тесты прошли успешно!")