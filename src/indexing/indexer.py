import os
import fitz 
import markdown2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

DATA_DIR = "data"
CHUNK_SIZE = 500
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/metadata.npy"

def read_pdf(file_path):
    doc = fitz.open(file_path)
    text_chunks = []
    metadata = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            text_chunks.append(chunk)
            metadata.append({
                "text": chunk, 
                "source": os.path.basename(file_path),
                "type": "pdf",
                "page": page_num,
                "chunk_id": idx
            })
    return text_chunks, metadata


def read_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = markdown2.markdown(f.read())
    chunks = chunk_text(text)
    metadata = [{
        "text": chunk,
        "source": os.path.basename(file_path),
        "type": "markdown",
        "chunk_id": i
    } for i, chunk in enumerate(chunks)]
    return chunks, metadata


def chunk_text(text, size=CHUNK_SIZE):
    text = text.strip().replace('\n', ' ')
    return [text[i:i+size] for i in range(0, len(text), size) if text[i:i+size].strip()]

def load_documents():
    all_chunks = []
    all_metadata = []

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)

        if filename.endswith(".pdf"):
            chunks, metadata = read_pdf(file_path)
        elif filename.endswith(".md"):
            chunks, metadata = read_md(file_path)
        else:
            continue

        all_chunks.extend(chunks)
        all_metadata.extend(metadata)

    return all_chunks, all_metadata

def index_documents():
    model = SentenceTransformer(EMBED_MODEL)
    documents, metadata = load_documents()

    print(f"Всего чанков для индексирования: {len(documents)}")
    embeddings = model.encode(documents, show_progress_bar=True)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)
    np.save(METADATA_PATH, metadata)

    print("Индекс сохранён в", INDEX_PATH)
    print("Метаданные сохранены в", METADATA_PATH)

if __name__ == "__main__":
    index_documents()
